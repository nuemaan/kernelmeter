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
from typing import Any, Callable

from . import attrs as _attrs
from . import peaks as _peaks
from .cudadrv import CudaNotAvailableError, Driver


@dataclass
class BenchSpec:
    fn: Callable
    name: str
    args: Callable[[], tuple]
    ref: Callable | None = None
    bytes_per_call: Callable[..., int] | int | None = None
    flops_per_call: Callable[..., int] | int | None = None
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
    pct_peak_bw: float | None = None
    pct_peak_fp32: float | None = None
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
    warmup: int = 10,
    iters: int = 100,
):
    """Register a function for ``kernelmeter bench``. Also usable directly:
    the decorated function is returned unchanged."""

    def deco(fn: Callable) -> Callable:
        REGISTRY.append(
            BenchSpec(
                fn=fn,
                name=name or fn.__name__,
                args=args,
                ref=ref,
                bytes_per_call=bytes_per_call,
                flops_per_call=flops_per_call,
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


def device_peaks() -> _peaks.Peaks:
    drv = Driver()
    dev = drv.device(0)
    return _peaks.derive(_attrs.query_all(drv, dev))


def run(spec: BenchSpec, peaks: _peaks.Peaks | None = None, flush_l2: bool = True) -> BenchResult:
    """Execute one spec on the current CUDA device."""
    if peaks is None:
        try:
            peaks = device_peaks()
        except CudaNotAvailableError:
            peaks = _peaks.Peaks(None, None, None)

    args = spec.args()

    correct = max_err = None
    if spec.ref is not None:
        correct, max_err = _check_correctness(spec, args)

    times = _time_fn(spec.fn, args, spec.warmup, spec.iters, flush_l2)
    ms_mean, ms_median, ms_min = summarize_times(times)

    nbytes = _resolve(spec.bytes_per_call, args)
    nflops = _resolve(spec.flops_per_call, args)
    gbps = achieved_gbps(nbytes, ms_median) if nbytes else None
    tflops = achieved_tflops(nflops, ms_median) if nflops else None

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
        pct_peak_bw=pct_of_peak(gbps, peaks.mem_bandwidth_gbs) if gbps else None,
        pct_peak_fp32=pct_of_peak(tflops, peaks.fp32_tflops) if tflops else None,
        ref_ms_median=ref_ms,
        speedup_vs_ref=speedup,
        correct=correct,
        max_abs_err=max_err,
    )


def run_registry(flush_l2: bool = True) -> list[BenchResult]:
    try:
        peaks = device_peaks()
    except CudaNotAvailableError:
        peaks = _peaks.Peaks(None, None, None)
    results = []
    for spec in REGISTRY:
        try:
            results.append(run(spec, peaks=peaks, flush_l2=flush_l2))
        except Exception as exc:  # surface per-kernel failures, keep going
            results.append(
                BenchResult(
                    name=spec.name, ms_mean=0.0, ms_median=0.0, ms_min=0.0, error=str(exc)
                )
            )
    return results
