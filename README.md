# kernelmeter

Two small tools for CUDA work, in one package with zero dependencies:

* `kernelmeter info` prints every device attribute your GPU driver knows,
  plus the card's theoretical peak bandwidth and FP32 throughput. No CUDA
  toolkit, no torch, no kernel launch. It reads straight from `libcuda`,
  which is part of the NVIDIA driver.
* `kernelmeter bench` times your kernel, checks the output against a
  reference, and tells you what fraction of the theoretical peak you hit.
  That last part is the point: 240 GB/s means nothing on its own, but
  "76% of what this card can physically do" tells you exactly how much
  room is left.

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
kernel                    median ms      GB/s  %peak bw   TFLOP/s   %fp32   vs ref  correct
-------------------------------------------------------------------------------------------
my_add                       3.3096     243.3     76.0%         -       -    1.02x     PASS
```

* **correct** - your output matched the reference. If this says FAIL,
  nothing else on the line matters.
* **%peak bw** - achieved bandwidth as a share of the card's theoretical
  maximum. For memory-bound kernels (most elementwise ops, reductions,
  softmax) this is the score to improve. Above ~80% there is little left
  to win.
* **%fp32** - same idea for arithmetic. Pass `flops_per_call` to get it.
  Chase this instead when your kernel is compute-bound.
* **vs ref** - speedup over the reference implementation.

Timing uses CUDA events with warmup, and the L2 cache is flushed between
iterations so small workloads can't fake huge bandwidth numbers from
cache hits. Pass `--no-flush-l2` if you want cache-hot numbers.

The [examples](examples/) folder has two ready-to-run Triton kernels
(vector add and a fused softmax) if you want a starting point.

## A workflow that works

If you are learning CUDA (say, working through the PMPP book) and wondering
whether your kernels are any good:

1. Run `kernelmeter info` once and note your card's two peak numbers.
2. Decide if your kernel is memory-bound or compute-bound.
3. Benchmark it. Memory-bound: look at `%peak bw`. Compute-bound: `%fp32`.
4. Under ~60%? Open the kernel in Nsight Compute. Now you know what you
   are looking for, instead of staring at forty unfamiliar counters.

## Caveats

* Peaks are computed from the max boost clock the driver reports. Sustained
  clocks under load are lower, so treat 85%+ as saturation.
* The FP32 peak is for plain CUDA cores. For tensor-core work, compare the
  reported TFLOP/s against your card's tensor peak yourself.
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

## License

MIT
