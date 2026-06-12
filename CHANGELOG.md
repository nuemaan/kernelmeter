# Changelog

## 0.2.0 - 2026-06-12

- bench scores kernels against the roofline: new `bound` and `%roof`
  columns, `peak_tflops=` override for tensor-core work
- `roofline` command: ridge point, attainable throughput and a terminal
  chart; works without a GPU via `--peak-bw`/`--peak-tflops`
- `occupancy` command: theoretical occupancy from block size, registers
  and shared memory, with a block-size sweep; covers CC 7.0 to 12.x
- `ceiling` command: measured STREAM bandwidth (copy/scale/add/triad)
  and a TF32-disabled fp32 matmul, reported against the theoretical peaks
- bench `--save`/`--compare` for regression tracking; exits non-zero
  when a kernel gets more than 5% slower than the baseline

## 0.1.0 - 2026-06-12

- `info`: every CUDA device attribute read straight from libcuda, no
  toolkit or kernel launch needed, plus derived peak bandwidth and fp32
  throughput; unknown attribute ids are probed and reported generically
- `bench`: CUDA-event timing with warmup and L2 flush, correctness check
  against a reference, achieved GB/s and TFLOP/s as a share of peak
