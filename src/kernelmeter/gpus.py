"""A small database of well-known GPUs, so the roofline math works for
cards you don't own.

This is what makes ``kernelmeter compare`` and ``kernelmeter roofline
--gpu 4090`` possible on a machine with no GPU at all: pick the cards,
get their ceilings, see which one your kernel actually wants.

Every entry stores physical parameters (SM count, boost clock, memory bus
and per-pin data rate) rather than headline numbers, and the peaks are
derived through the same formulas used for live devices. The test suite
asserts each derived value against the published spec sheet, so a wrong
entry fails CI rather than quietly misleading someone.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import peaks as _peaks


@dataclass(frozen=True)
class GpuSpec:
    id: str
    name: str
    cc: tuple[int, int]
    sm_count: int
    boost_mhz: int
    mem_bus_bits: int
    mem_gbps: float  # effective per-pin data rate
    tdp_w: int
    vram_gb: int

    def peaks(self) -> _peaks.Peaks:
        clock_khz = self.boost_mhz * 1000
        return _peaks.Peaks(
            mem_bandwidth_gbs=self.mem_gbps * self.mem_bus_bits / 8.0,
            fp32_tflops=_peaks.fp32_tflops(self.sm_count, clock_khz, *self.cc),
            compute_capability=self.cc,
            fp16_tensor_tflops=_peaks.fp16_tensor_tflops(self.sm_count, clock_khz, *self.cc),
            tf32_tensor_tflops=_peaks.tf32_tensor_tflops(self.sm_count, clock_khz, *self.cc),
        )


# Physical parameters from vendor spec sheets. Peaks derived, not copied,
# so any typo here shows up as a spec-sheet mismatch in tests.
DATABASE: tuple[GpuSpec, ...] = (
    GpuSpec("t4", "Tesla T4", (7, 5), 40, 1590, 256, 10.0, 70, 16),
    GpuSpec("v100-sxm2", "Tesla V100 SXM2", (7, 0), 80, 1530, 4096, 1.758, 300, 16),
    GpuSpec("v100-pcie", "Tesla V100 PCIe", (7, 0), 80, 1380, 4096, 1.758, 250, 16),
    GpuSpec("a30", "A30", (8, 0), 56, 1440, 3072, 2.43, 165, 24),
    GpuSpec("a100-40gb", "A100 SXM4 40GB", (8, 0), 108, 1410, 5120, 2.43, 400, 40),
    GpuSpec("a100-80gb", "A100 SXM4 80GB", (8, 0), 108, 1410, 5120, 3.186, 400, 80),
    GpuSpec("h100-sxm", "H100 SXM5", (9, 0), 132, 1980, 5120, 5.23, 700, 80),
    GpuSpec("h100-pcie", "H100 PCIe", (9, 0), 114, 1755, 5120, 3.13, 350, 80),
    GpuSpec("a10", "A10", (8, 6), 72, 1695, 384, 12.5, 150, 24),
    GpuSpec("a40", "A40", (8, 6), 84, 1740, 384, 14.5, 300, 48),
    GpuSpec("l4", "L4", (8, 9), 58, 2040, 192, 12.5, 72, 24),
    GpuSpec("l40", "L40", (8, 9), 142, 2490, 384, 18.0, 300, 48),
    GpuSpec("l40s", "L40S", (8, 9), 142, 2520, 384, 18.0, 350, 48),
    GpuSpec("rtx-a4000", "RTX A4000", (8, 6), 48, 1560, 256, 14.0, 140, 16),
    GpuSpec("rtx-a5000", "RTX A5000", (8, 6), 64, 1695, 384, 16.0, 230, 24),
    GpuSpec("rtx-a6000", "RTX A6000", (8, 6), 84, 1800, 384, 16.0, 300, 48),
    GpuSpec("rtx-6000-ada", "RTX 6000 Ada", (8, 9), 142, 2505, 384, 20.0, 300, 48),
    GpuSpec("titan-rtx", "Titan RTX", (7, 5), 72, 1770, 384, 14.0, 280, 24),
    GpuSpec("rtx-2080-ti", "GeForce RTX 2080 Ti", (7, 5), 68, 1545, 352, 14.0, 250, 11),
    GpuSpec("rtx-3060", "GeForce RTX 3060", (8, 6), 28, 1777, 192, 15.0, 170, 12),
    GpuSpec("rtx-3070", "GeForce RTX 3070", (8, 6), 46, 1725, 256, 14.0, 220, 8),
    GpuSpec("rtx-3080", "GeForce RTX 3080", (8, 6), 68, 1710, 320, 19.0, 320, 10),
    GpuSpec("rtx-3080-ti", "GeForce RTX 3080 Ti", (8, 6), 80, 1665, 384, 19.0, 350, 12),
    GpuSpec("rtx-3090", "GeForce RTX 3090", (8, 6), 82, 1695, 384, 19.5, 350, 24),
    GpuSpec("rtx-3090-ti", "GeForce RTX 3090 Ti", (8, 6), 84, 1860, 384, 21.0, 450, 24),
    GpuSpec("rtx-4080", "GeForce RTX 4080", (8, 9), 76, 2505, 256, 22.4, 320, 16),
    GpuSpec("rtx-4090", "GeForce RTX 4090", (8, 9), 128, 2520, 384, 21.0, 450, 24),
    GpuSpec("rtx-5070", "GeForce RTX 5070", (12, 0), 48, 2512, 192, 28.0, 250, 12),
    GpuSpec("rtx-5070-ti", "GeForce RTX 5070 Ti", (12, 0), 70, 2452, 256, 28.0, 300, 16),
    GpuSpec("rtx-5080", "GeForce RTX 5080", (12, 0), 84, 2617, 256, 30.0, 360, 16),
    GpuSpec("rtx-5090", "GeForce RTX 5090", (12, 0), 170, 2407, 512, 28.0, 575, 32),
)

_BY_ID = {spec.id: spec for spec in DATABASE}


class UnknownGpuError(ValueError):
    pass


def resolve(query: str) -> GpuSpec:
    """Match a user string against the database. Exact id first, then
    'rtx-' + query ('3090' -> rtx-3090, not the ti), then substring.
    Remaining ambiguity is an error that lists the candidates instead of
    guessing."""
    q = query.lower().strip()
    if q in _BY_ID:
        return _BY_ID[q]
    if f"rtx-{q}" in _BY_ID:
        return _BY_ID[f"rtx-{q}"]
    matches = [s for s in DATABASE if q in s.id or q in s.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise UnknownGpuError(
            f"unknown gpu {query!r}. Run 'kernelmeter gpus' for the list."
        )
    ids = ", ".join(s.id for s in matches)
    raise UnknownGpuError(f"{query!r} is ambiguous: {ids}")
