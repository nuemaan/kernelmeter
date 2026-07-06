"""Measure what the card actually delivers, not what the spec sheet says.

Theoretical peaks are computed from max boost clocks and are never fully
reachable. This module runs the four STREAM kernels (copy, scale, add,
triad) through torch to find the real bandwidth ceiling, and a large
TF32-disabled matmul to find the real FP32 ceiling. Judge your kernels
against these numbers and 100% actually means 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import bench as _bench
from . import peaks as _peaks


@dataclass
class CeilingResult:
    name: str
    ms: float
    gbps: float | None = None
    tflops: float | None = None
    pct_theoretical: float | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _stream_specs(n: int):
    """The four STREAM kernels and the bytes each one moves."""
    import torch

    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    q = 2.5
    elt = a.element_size()
    return [
        ("copy", lambda: b.copy_(a), 2 * n * elt),
        ("scale", lambda: torch.mul(a, q, out=b), 2 * n * elt),
        ("add", lambda: torch.add(a, b, out=c), 3 * n * elt),
        ("triad", lambda: torch.add(b, c, alpha=q, out=a), 3 * n * elt),
    ]


def measure(
    mb: int = 256, matmul_n: int = 4096, warmup: int = 10, iters: int = 50,
    device_index: int = 0,
) -> list[CeilingResult]:
    import torch

    if device_index:
        torch.cuda.set_device(device_index)
    try:
        device_peaks = _bench.device_peaks(device_index)
    except Exception:
        device_peaks = _peaks.Peaks(None, None, None)

    n = mb * 1024 * 1024 // 4  # fp32 elements
    results = []
    for name, fn, nbytes in _stream_specs(n):
        times = _bench._time_fn(lambda *_: fn(), (), warmup, iters, flush_l2=True)
        ms = _bench.summarize_times(times)[1]
        gbps = _bench.achieved_gbps(nbytes, ms)
        results.append(
            CeilingResult(
                name=name,
                ms=ms,
                gbps=gbps,
                pct_theoretical=_bench.pct_of_peak(gbps, device_peaks.mem_bandwidth_gbs),
            )
        )

    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # keep it on the fp32 units
    try:
        m = torch.randn(matmul_n, matmul_n, device="cuda")
        out = torch.empty_like(m)
        times = _bench._time_fn(
            lambda *_: torch.mm(m, m, out=out), (), warmup, iters, flush_l2=True
        )
        ms = _bench.summarize_times(times)[1]
        tflops = _bench.achieved_tflops(2 * matmul_n**3, ms)
        results.append(
            CeilingResult(
                name="fp32 matmul",
                ms=ms,
                tflops=tflops,
                pct_theoretical=_bench.pct_of_peak(tflops, device_peaks.fp32_tflops),
            )
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    return results


def format_table(results: list[CeilingResult]) -> list[str]:
    header = f"{'test':<14} {'median ms':>10} {'GB/s':>9} {'TFLOP/s':>9} {'% of theoretical':>17}"
    lines = [header, "-" * len(header)]
    for r in results:
        gbps = f"{r.gbps:.1f}" if r.gbps is not None else "-"
        tflops = f"{r.tflops:.2f}" if r.tflops is not None else "-"
        pct = f"{r.pct_theoretical:.1f}%" if r.pct_theoretical is not None else "-"
        lines.append(f"{r.name:<14} {r.ms:>10.4f} {gbps:>9} {tflops:>9} {pct:>17}")
    best_bw = max((r.gbps for r in results if r.gbps), default=None)
    if best_bw:
        lines.append("")
        lines.append(
            f"measured bandwidth ceiling: {best_bw:.1f} GB/s "
            "(use this as the honest 100% for memory-bound kernels)"
        )
    return lines
