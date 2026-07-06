"""Benchmark kernels against references and against the hardware ceiling.

Workflow: decorate your implementation with @kernelmeter.benchmark, giving
it an argument factory, a reference implementation, and (optionally) the
bytes moved / FLOPs performed per call. Run ``kernelmeter bench yourfile.py``
and you get latency, achieved GB/s and TFLOP/s, percent of the device's
theoretical peak, and a numerical correctness check against the reference.

torch is imported lazily so the rest of the package (``kernelmeter info``)
works without it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from . import attrs as _attrs
from . import peaks as _peaks
from . import roofline as _roofline
from .cudadrv import CudaNotAvailableError, Driver


@dataclass
class BenchSpec:
    fn: Callable
    name: str
    args: Callable[[], tuple]
    ref: Callable | None = None
    bytes_per_call: Callable[..., int] | int | None = None
    flops_per_call: Callable[..., int] | int | None = None
    peak_tflops: float | None = None  # override for tensor-core work
    warmup: int = 10
    iters: int = 100


@dataclass
class BenchResult:
    name: str
    ms_mean: float
    ms_median: float
    ms_min: float
    gbps: float | None = None
    tflops: float | None = None
    intensity: float | None = None
    bound: str | None = None
    pct_roofline: float | None = None
    pct_roof_sustained: float | None = None
    pct_peak_bw: float | None = None
    pct_peak_fp32: float | None = None
    sm_clock_mhz: float | None = None
    max_sm_clock_mhz: int | None = None
    mem_clock_mhz: float | None = None
    temperature_c: int | None = None
    power_w: float | None = None
    ref_ms_median: float | None = None
    speedup_vs_ref: float | None = None
    correct: bool | None = None
    max_abs_err: float | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


REGISTRY: list[BenchSpec] = []


def benchmark(
    name: str | None = None,
    *,
    args: Callable[[], tuple],
    ref: Callable | None = None,
    bytes_per_call: Callable[..., int] | int | None = None,
    flops_per_call: Callable[..., int] | int | None = None,
    peak_tflops: float | None = None,
    warmup: int = 10,
    iters: int = 100,
):
    """Register a function for ``kernelmeter bench``. Also usable directly:
    the decorated function is returned unchanged.

    peak_tflops replaces the derived fp32 peak in the roofline when your
    kernel runs on other units (tensor cores, fp16, fp64).
    """

    def deco(fn: Callable) -> Callable:
        REGISTRY.append(
            BenchSpec(
                fn=fn,
                name=name or fn.__name__,
                args=args,
                ref=ref,
                bytes_per_call=bytes_per_call,
                flops_per_call=flops_per_call,
                peak_tflops=peak_tflops,
                warmup=warmup,
                iters=iters,
            )
        )
        return fn

    return deco


# ---------------------------------------------------------------------------
# Pure math helpers (unit-tested without a GPU)
# ---------------------------------------------------------------------------

def summarize_times(times_ms: list[float]) -> tuple[float, float, float]:
    return (
        statistics.fmean(times_ms),
        statistics.median(times_ms),
        min(times_ms),
    )


def achieved_gbps(bytes_per_call: int, ms: float) -> float:
    return bytes_per_call / (ms * 1e-3) / 1e9


def achieved_tflops(flops_per_call: int, ms: float) -> float:
    return flops_per_call / (ms * 1e-3) / 1e12


def pct_of_peak(achieved: float, peak: float | None) -> float | None:
    if not peak:
        return None
    return 100.0 * achieved / peak


def _resolve(metric: Callable[..., int] | int | None, args: tuple) -> int | None:
    if metric is None:
        return None
    if callable(metric):
        return metric(*args)
    return int(metric)


def roofline_score(
    nbytes: int | None,
    nflops: int | None,
    gbps: float | None,
    tflops: float | None,
    peaks: _peaks.Peaks,
    peak_tflops_override: float | None = None,
) -> tuple[float | None, str | None, float | None]:
    """(intensity, bound, pct of attainable) given whatever metrics exist.

    With only bytes the kernel is treated as memory-bound, with only flops
    as compute-bound. With both, the roofline decides.
    """
    peak_tf = peak_tflops_override or peaks.fp32_tflops
    peak_bw = peaks.mem_bandwidth_gbs

    if nbytes and nflops and peak_tf and peak_bw and tflops:
        ai = _roofline.intensity(nflops, nbytes)
        attainable = _roofline.attainable_tflops(ai, peak_tf, peak_bw)
        return ai, _roofline.bound(ai, peak_tf, peak_bw), 100.0 * tflops / attainable
    if nbytes and gbps and peak_bw:
        return None, "mem", 100.0 * gbps / peak_bw
    if nflops and tflops and peak_tf:
        return None, "comp", 100.0 * tflops / peak_tf
    return None, None, None


def sustained_peaks(peaks: _peaks.Peaks, telemetry, peak_tflops_override: float | None = None) -> _peaks.Peaks:
    """Scale the theoretical peaks down to the clocks the card actually
    held while the kernel ran."""
    tf = peak_tflops_override or peaks.fp32_tflops
    return _peaks.Peaks(
        mem_bandwidth_gbs=(
            peaks.mem_bandwidth_gbs * telemetry.mem_clock_fraction
            if peaks.mem_bandwidth_gbs
            else None
        ),
        fp32_tflops=tf * telemetry.sm_clock_fraction if tf else None,
        compute_capability=peaks.compute_capability,
    )


def diff_results(baseline: list[dict], results: list["BenchResult"], threshold_pct: float = 5.0):
    """Compare a run against a saved baseline. Returns (rows, regressions)
    where rows are (name, old_ms, new_ms, delta_pct) and regressions lists
    names that got slower by more than the threshold."""
    old = {r["name"]: r for r in baseline}
    rows = []
    regressions = []
    for r in results:
        if r.error or r.name not in old:
            continue
        old_ms = old[r.name]["ms_median"]
        delta = 100.0 * (r.ms_median - old_ms) / old_ms if old_ms else 0.0
        rows.append((r.name, old_ms, r.ms_median, delta))
        if delta > threshold_pct:
            regressions.append(r.name)
    return rows, regressions


# ---------------------------------------------------------------------------
# GPU execution (requires torch + an NVIDIA device)
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable, args: tuple, warmup: int, iters: int, flush_l2: bool) -> list[float]:
    import torch

    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Flushing L2 between iterations keeps memory-bound numbers honest:
    # without it, small workloads get served from cache and report
    # impossible bandwidths. Same approach as triton.testing.do_bench.
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda") if flush_l2 else None

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cache is not None:
            cache.zero_()
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _check_correctness(spec: BenchSpec, args: tuple) -> tuple[bool, float]:
    import torch

    got = spec.fn(*args)
    want = spec.ref(*args)
    max_err = (got.float() - want.float()).abs().max().item()
    try:
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-3, check_dtype=False)
        return True, max_err
    except AssertionError:
        return False, max_err


def device_peaks(device_index: int = 0) -> _peaks.Peaks:
    drv = Driver()
    dev = drv.device(device_index)
    return _peaks.derive(_attrs.query_all(drv, dev))


def run(
    spec: BenchSpec,
    peaks: _peaks.Peaks | None = None,
    flush_l2: bool = True,
    device_index: int = 0,
) -> BenchResult:
    """Execute one spec on the current CUDA device."""
    if peaks is None:
        try:
            peaks = device_peaks(device_index)
        except CudaNotAvailableError:
            peaks = _peaks.Peaks(None, None, None)

    args = spec.args()

    correct = max_err = None
    if spec.ref is not None:
        correct, max_err = _check_correctness(spec, args)

    monitor = None
    try:
        from . import nvml as _nvml

        monitor = _nvml.Monitor(device_index=device_index)
        monitor.start()
    except Exception:
        monitor = None

    times = _time_fn(spec.fn, args, spec.warmup, spec.iters, flush_l2)

    telemetry = None
    if monitor is not None:
        try:
            telemetry = monitor.stop()
            monitor.close()
        except Exception:
            telemetry = None

    ms_mean, ms_median, ms_min = summarize_times(times)

    nbytes = _resolve(spec.bytes_per_call, args)
    nflops = _resolve(spec.flops_per_call, args)
    gbps = achieved_gbps(nbytes, ms_median) if nbytes else None
    tflops = achieved_tflops(nflops, ms_median) if nflops else None
    ai, kernel_bound, pct_roof = roofline_score(
        nbytes, nflops, gbps, tflops, peaks, spec.peak_tflops
    )

    pct_sustained = None
    if telemetry is not None:
        scaled = sustained_peaks(peaks, telemetry, spec.peak_tflops)
        pct_sustained = roofline_score(nbytes, nflops, gbps, tflops, scaled)[2]

    ref_ms = speedup = None
    if spec.ref is not None:
        ref_times = _time_fn(spec.ref, args, spec.warmup, spec.iters, flush_l2)
        ref_ms = summarize_times(ref_times)[1]
        speedup = ref_ms / ms_median if ms_median > 0 else None

    return BenchResult(
        name=spec.name,
        ms_mean=ms_mean,
        ms_median=ms_median,
        ms_min=ms_min,
        gbps=gbps,
        tflops=tflops,
        intensity=ai,
        bound=kernel_bound,
        pct_roofline=pct_roof,
        pct_roof_sustained=pct_sustained,
        pct_peak_bw=pct_of_peak(gbps, peaks.mem_bandwidth_gbs) if gbps else None,
        pct_peak_fp32=pct_of_peak(tflops, peaks.fp32_tflops) if tflops else None,
        sm_clock_mhz=telemetry.sm_clock_mhz if telemetry else None,
        max_sm_clock_mhz=telemetry.max_sm_clock_mhz if telemetry else None,
        mem_clock_mhz=telemetry.mem_clock_mhz if telemetry else None,
        temperature_c=telemetry.temperature_c if telemetry else None,
        power_w=telemetry.power_w if telemetry else None,
        ref_ms_median=ref_ms,
        speedup_vs_ref=speedup,
        correct=correct,
        max_abs_err=max_err,
    )


def run_registry(flush_l2: bool = True, device_index: int = 0) -> list[BenchResult]:
    if device_index:
        import torch

        torch.cuda.set_device(device_index)
    try:
        peaks = device_peaks(device_index)
    except CudaNotAvailableError:
        peaks = _peaks.Peaks(None, None, None)
    results = []
    for spec in REGISTRY:
        try:
            results.append(run(spec, peaks=peaks, flush_l2=flush_l2, device_index=device_index))
        except Exception as exc:  # surface per-kernel failures, keep going
            results.append(
                BenchResult(
                    name=spec.name, ms_mean=0.0, ms_median=0.0, ms_min=0.0, error=str(exc)
                )
            )
    return results
