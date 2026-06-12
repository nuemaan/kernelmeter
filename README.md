# kernelmeter

[![PyPI](https://img.shields.io/pypi/v/kernelmeter)](https://pypi.org/project/kernelmeter/)
[![CI](https://github.com/nuemaan/kernelmeter/actions/workflows/ci.yml/badge.svg)](https://github.com/nuemaan/kernelmeter/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/kernelmeter)](https://pypi.org/project/kernelmeter/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Small tools for one question: **is my GPU kernel actually good, and if
not, what exactly is holding it back?** All in one package with zero
required dependencies.

* `kernelmeter info` prints every device attribute your GPU driver knows,
  plus the card's theoretical peak bandwidth and FP32 throughput. No CUDA
  toolkit, no torch, no kernel launch. It reads straight from `libcuda`,
  which is part of the NVIDIA driver.
* `kernelmeter bench` times your kernel, checks the output against a
  reference, and scores it against the roofline: the best your card could
  possibly do for that kernel's mix of math and memory traffic. 240 GB/s
  means nothing on its own; "76% of attainable" tells you how much room
  is left.
* `kernelmeter roofline` draws your card's roofline in the terminal and
  shows where a kernel sits on it.
* `kernelmeter occupancy` answers "why is my occupancy 50%?" from block
  size, registers and shared memory, and shows which block sizes fix it.
* `kernelmeter ceiling` measures what the card *really* delivers
  (STREAM-style bandwidth tests plus a big FP32 matmul), because spec
  sheet numbers are never fully reachable.

## Install

```bash
pip install kernelmeter           # info only, no dependencies
pip install "kernelmeter[bench]"  # adds torch for the bench harness
```

Or from source:

```bash
git clone https://github.com/nuemaan/kernelmeter
cd kernelmeter
pip install -e ".[bench]"
```

## Querying your GPU

```bash
kernelmeter info
```

Output from a Tesla T4:

```text
CUDA driver version : 13.0

Device 0: Tesla T4 (14.6 GiB)
  compute capability        : 7.5
  theoretical mem bandwidth : 320.1 GB/s
  theoretical FP32 peak     : 8.14 TFLOP/s

  attribute                                        value
  ------------------------------------------------ ------------
  max_threads_per_block                            1024
  max_block_dim_x                                  1024
  max_shared_memory_per_block                      49152
  warp_size                                        32
  clock_rate_khz                                   1590000
  ...                                              (147 attributes total)
```

These are the same values Nsight Compute shows as `device__attribute_*`,
except you don't need to profile a kernel to see them. Add `--json` for
machine-readable output.

Every attribute id is probed against the live driver, so the output always
matches the machine you run it on. Ids newer than the bundled name table
still show up, just under a generic `attribute_<id>` name.

## Benchmarking a kernel

Three steps.

**1. Write your kernel in a file and decorate it.** Anything callable from
Python works: Triton kernels, custom CUDA extensions, `torch.compile`
output, CuPy. Here is a complete file you can copy:

```python
# mybench.py
import torch
import kernelmeter as km

N = 1 << 26  # work on big inputs so you measure memory, not cache

def make_args():
    return (torch.randn(N, device="cuda"), torch.randn(N, device="cuda"))

@km.benchmark(
    "my_add",
    args=make_args,                 # builds fresh inputs for the run
    ref=torch.add,                  # trusted implementation to compare with
    bytes_per_call=lambda x, y: 3 * x.numel() * x.element_size(),
)
def my_add(x, y):
    return x + y                    # <- replace with your kernel
```

`bytes_per_call` is how much memory the algorithm has to move (here: read
x, read y, write the result). The tool divides it by measured time to get
your effective bandwidth.

**2. Run it.**

```bash
kernelmeter bench mybench.py
```

**3. Read the result.** From a T4, with the add written as a Triton kernel:

```text
kernel                    median ms      GB/s   TFLOP/s  bound    %roof   vs ref  correct
------------------------------------------------------------------------------------------
my_add                       3.3393     241.2         -    mem    75.3%    1.01x     PASS
```

* **correct** - your output matched the reference. If this says FAIL,
  nothing else on the line matters.
* **bound** - whether the memory system (`mem`) or the ALUs (`comp`) limit
  this kernel, decided by its arithmetic intensity (flops per byte).
* **%roof** - how close you are to the best this card could possibly do
  for that intensity. This is the score to improve. Above ~80% there is
  little left to win.
* **vs ref** - speedup over the reference implementation.

Pass `flops_per_call` too and the roofline model places your kernel
precisely; pass `peak_tflops=...` if your kernel runs on tensor cores so
it gets judged against the right ceiling. Raw `%peak bw` and `%fp32`
numbers are always in the `--json` output.

Timing uses CUDA events with warmup, and the L2 cache is flushed between
iterations so small workloads can't fake huge bandwidth numbers from
cache hits. Pass `--no-flush-l2` if you want cache-hot numbers.

The [examples](examples/) folder has ready-to-run starting points: two
Triton kernels (vector add, fused softmax) and a compute-bound matmul.

## Seeing the roofline

```bash
kernelmeter roofline --ai 0.33        # mark a kernel at 0.33 flop/byte
```

```text
  peak bandwidth : 320.0 GB/s
  peak compute   : 8.14 TFLOP/s
  ridge point    : 25.4 flop/byte

8.14 TF/s |                                      **x*****************
          |                                   ***
          |                               ****
          |                            ***
          |                        ****
          |                    ****
          |                 ***
          |             ****
          |          ***
          |      *o**
          |  ****
          |**
          +----------------------------------------------------------
           2^-3            2^0            2^3             2^6

at 0.33 flop/byte the kernel is memory-bound; attainable: 0.11 TFLOP/s
```

The `o` is your kernel, the `x` is the ridge point. Left of the ridge,
more FLOPs are free: the memory traffic is the bill you are paying
anyway. That is the whole argument for kernel fusion, in one picture.
No GPU around? `--peak-bw` and `--peak-tflops` let you draw any card.

## Why is my occupancy low?

Feed it what `ptxas -v` or Nsight Compute tells you about your kernel:

```bash
kernelmeter occupancy --block 256 --regs 64 --smem 8192 --cc 8.6
```

```text
occupancy for compute capability 8.6
  block=256 regs/thread=64 smem/block=8192

  occupancy    : 66.7% (32/48 warps per SM)
  blocks per SM: 4
  limited by   : registers

  block size      64    128    192    256    384    512    768   1024
  occupancy      46%    67%    62%    67%    50%    67%    50%    67%
```

It names the resource that is capping you and sweeps block sizes so you
can see if a different launch shape helps. Works with no GPU present:
pass `--cc` for any architecture from 7.0 (Volta) to 12.x (Blackwell).

## What can the card really do?

Theoretical peaks assume the max boost clock, which the card cannot hold.
Measure the real ceilings once and judge your kernels against those:

```bash
kernelmeter ceiling
```

This runs the four STREAM kernels (copy, scale, add, triad) and a large
TF32-disabled matmul. On the same T4:

```text
test            median ms      GB/s   TFLOP/s  % of theoretical
---------------------------------------------------------------
copy               1.1495     233.5         -             73.0%
scale              1.1674     230.0         -             71.8%
add                1.6903     238.2         -             74.4%
triad              1.6878     238.6         -             74.5%
fp32 matmul        3.5563         -      4.83             59.3%

measured bandwidth ceiling: 238.6 GB/s (use this as the honest 100%
for memory-bound kernels)
```

This reframes the bench results above: the vector add that scored "75.3%
of theoretical" was moving 241 GB/s on a card whose memory system tops
out at 238.6 GB/s in practice. It was already saturated. Without the
measured ceiling you would have kept optimizing a finished kernel.

## Catching regressions

```bash
kernelmeter bench mykernels.py --save baseline.json
# ...edit your kernels...
kernelmeter bench mykernels.py --compare baseline.json
```

The compare run prints a delta column per kernel and exits non-zero if
anything got more than 5% slower, so it slots straight into CI.

## A workflow that works

If you are learning CUDA (say, working through the PMPP book) and wondering
whether your kernels are any good:

1. Run `kernelmeter info` and `kernelmeter ceiling` once. Now you know
   your card's real limits.
2. Benchmark your kernel with `bytes_per_call` and `flops_per_call` set.
   The `bound` column tells you which resource you are fighting.
3. `%roof` under ~60%? If the kernel is memory-bound, check `occupancy`
   first: too few warps in flight cannot hide memory latency. Then open
   Nsight Compute. Now you know what you are looking for, instead of
   staring at forty unfamiliar counters.
4. `%roof` above ~80%? Stop optimizing this kernel. The next win is
   algorithmic (fuse it with a neighbor, move less data), and the
   roofline chart shows why: left of the ridge, FLOPs are free.

## Caveats

* Theoretical peaks are computed from the max boost clock the driver
  reports. Sustained clocks under load are lower; `kernelmeter ceiling`
  measures what you can actually reach.
* The derived compute peak is for plain FP32 on CUDA cores. For
  tensor-core kernels pass `peak_tflops=...` to the benchmark decorator
  so the roofline uses the right roof.
* The occupancy command implements the standard calculator model. Real
  occupancy can differ (launch bounds, driver decisions); confirm with
  Nsight Compute when it matters.
* Attribute names above id 121 are best-effort against the CUDA 12.x
  headers. Values are always read live from your driver. PRs that extend
  the name table are welcome.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests fake the driver, so they run anywhere, no GPU needed. CI runs
them on plain GitHub runners. For an end-to-end check on a real GPU there
is a [Modal](https://modal.com) script: `modal run scripts/modal_gpu_test.py`.
The numbers in this README come from that script on a T4.

Releases are tag-driven: bump the version in `pyproject.toml`, add a
[CHANGELOG.md](CHANGELOG.md) entry, push a `v*` tag. CI tests, builds and
publishes to PyPI through trusted publishing.

## License

MIT
