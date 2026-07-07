# Changelog

## 0.8.0 - 2026-07-07

- amd support in the database: mi100, mi210, mi250x, mi300x, rx 6900 xt,
  rx 7900 xt/xtx, rx 9070 xt, radeon pro w7900. entries store compute
  units and clocks, peaks derive through per-architecture rate tables
  (cdna1-3, rdna2-4), and every derived number is asserted against the
  vendor sheet like the nvidia entries. 40 cards total.
- compare, llm, roofline --gpu and report --gpu treat vendors equally,
  so cross-vendor questions like "one mi300x or one h100 for a 70b" are
  now a one-liner. live-device commands (info, bench, ceiling) still
  need cuda; rocm device support wants someone with the hardware.
- the fp16 column for amd is matrix-core throughput on cdna and rdna3+,
  packed vector math on rdna2. tf32 stays nvidia-only. the geforce
  fp32-accumulate halving does not apply to radeon matrix rates.
- gpus lists an arch column (compute capability for nvidia, cdna3/rdna4
  style names for amd).

## 0.7.1 - 2026-07-07

- fixed two crashes found in an edge-case audit: occupancy with an
  out-of-range block size and llm with --num-gpus 0 both crashed with
  tracebacks instead of printing clean errors. both exit 1 with a message
  now, and llm.estimate() validates its inputs.
- local titan cards are reported by the driver in caps ("NVIDIA TITAN
  RTX"), which dodged the consumer-card check, so their prefill ceiling
  wasn't halved. detection is case-insensitive now.
- report prints a clean error when the output path isn't writable.
- llm added to the cli help header.

## 0.7.0 - 2026-07-07

- llm: geforce/titan prefill ceilings now use the fp32-accumulate tensor
  rate (half the fp16 marketing number), which is what inference stacks
  actually run. datacenter cards are unchanged.
- llm: --num-gpus splits weights over identical cards (2x3090 for a 70b),
  --active-params handles moe models (fit needs all weights, decode reads
  only the active ones), --batch shows total and per-stream throughput
  ceilings, --per-watt adds tokens per watt from tdp.
- bench and ceiling take --device, and the nvml telemetry monitor now
  follows it: on multi-gpu machines the telemetry table used to read
  device 0 regardless of where the kernels ran.
- occupancy notes when a compute capability's limits are borrowed from
  the nearest known architecture instead of failing silently.
- ships py.typed, adds CONTRIBUTING.md.

## 0.6.0 - 2026-07-06

- new `llm` command: token-throughput ceilings for llm inference from the
  same roofline math. decode is memory-bound (every token reads all the
  weights), prefill is compute-bound (~2 flops per param per token), so
  the ceilings need no benchmark. checks vram fit, takes --quant presets
  with gguf-realistic bytes per parameter or an exact --bytes-per-param,
  and --cost adds tokens-per-second per rental dollar. works against the
  card database or the local device.
- the database now carries vram capacity for every card (spec-sheet
  tested like the rest); `gpus` grew a vram column.

## 0.5.1 - 2026-07-06

- 11 more cards in the database (v100-pcie, a30, l40, rtx a4000/a5000,
  titan rtx, 2080 ti, 3080 ti, 3090 ti, 5070, 5070 ti), each spec-sheet
  asserted in tests like the rest. 31 cards total.
- resolve() prefers the exact card over ti variants, so "3090" still
  means rtx-3090 now that rtx-3090-ti exists.
- scripts/update_webdb.py re-embeds the database into the web demo;
  the drift test tells you when you forgot.

## 0.5.0 - 2026-06-14

- built-in database of 20 well-known GPUs (T4 through RTX 5090, A100/H100
  and the workstation cards). Entries store physical parameters and derive
  their peaks through the same formulas used for live devices; the test
  suite asserts every derived number against the vendor spec sheet.
- `gpus` lists the database, `compare` puts cards side by side with an
  overlaid multi-card roofline, and `--ai` scores them at your kernel's
  intensity. `--cost 4090=0.35,...` adds an attainable-TFLOPs-per-dollar
  column for rental decisions.
- `roofline --gpu 4090` draws any database card without owning it.
- `report` writes a single-file HTML report card (SVG roofline, peak
  tiles, NVML facts, launch limits) for the local device or any database
  card. No javascript, no external assets.

## 0.4.2 - 2026-06-14

- nvml brand map now covers the 12-16 range (Quadro RTX, NVIDIA RTX,
  NVIDIA, GeForce RTX, Titan RTX). Modern data-center cards report
  NVML_BRAND_NVIDIA (14), so a current Tesla/A/H-series card used to come
  back with an empty brand; it now resolves (verified "NVIDIA" on a T4).

## 0.4.1 - 2026-06-14

- named device attributes 149-154 (d3d12_cig_streams_supported,
  dma_buf_mmap_supported, and the logical_endpoint_* family), verified
  against the CUDA 13.x driver enum. CU_DEVICE_ATTRIBUTE_MAX moves up with
  each toolkit release, so the table now tracks the latest defined ids and
  the probe still surfaces anything newer as attribute_<id>.

## 0.4.0 - 2026-06-13

- `info` now adds an NVML section with facts the driver attribute enum
  doesn't expose: architecture name, real CUDA core count, PCIe link
  (current/max gen and width), live memory in use, ECC state, VBIOS
  version. NVML ships with the driver, so still no toolkit dependency.
  Present in the json output under `devices[].nvml`; skipped silently
  when NVML isn't available.
- new `kernelmeter.extras` module and `DeviceExtras` for programmatic use.

## 0.3.1 - 2026-06-13

- named device attributes 144-148 (host_memory_pools_supported,
  host_virtual_memory_management_supported, host_alloc_dma_buf_supported,
  only_partial_host_native_atomic_supported, atomic_reduction_supported),
  covering the driver enum up to the CU_DEVICE_ATTRIBUTE_MAX sentinel

## 0.3.0 - 2026-06-12

- live telemetry through NVML: bench samples SM/memory clocks,
  temperature and power while each kernel runs, and reports `%roof@clk`,
  the roofline score against the ceiling the card actually held
- `info` shows current clocks, temperature and power when NVML is present
- derived fp16 and tf32 tensor-core peaks per architecture (Volta through
  Blackwell); `roofline --tensor` draws the tensor roof
- named device attributes 122-143 (dma_buf_supported, numa_id,
  multicast_supported, gpu_pci_device_id, ...) so CUDA 12.x/13.x drivers
  report fewer anonymous ids

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
