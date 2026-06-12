"""Derive theoretical peak throughput ("speed of light") from device attributes.

These numbers are upper bounds computed from the max boost clock the driver
reports; sustained clocks under load are usually lower, so treat a kernel
at 85%+ of these peaks as effectively saturating the machine.
"""

from __future__ import annotations

from dataclasses import dataclass

# FP32 CUDA cores per SM, keyed by compute capability. The fallback for an
# unknown capability is the value of the closest older architecture.
_FP32_CORES_PER_SM: dict[tuple[int, int], int] = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128, (10, 3): 128,
    (12, 0): 128, (12, 1): 128,
}


def fp32_cores_per_sm(major: int, minor: int) -> int:
    if (major, minor) in _FP32_CORES_PER_SM:
        return _FP32_CORES_PER_SM[(major, minor)]
    older = [cc for cc in _FP32_CORES_PER_SM if cc <= (major, minor)]
    if older:
        return _FP32_CORES_PER_SM[max(older)]
    return 64


@dataclass
class Peaks:
    """Theoretical per-device ceilings derived from driver attributes."""

    mem_bandwidth_gbs: float | None
    fp32_tflops: float | None
    compute_capability: tuple[int, int] | None

    def as_dict(self) -> dict:
        return {
            "theoretical_mem_bandwidth_gb_s": self.mem_bandwidth_gbs,
            "theoretical_fp32_tflops": self.fp32_tflops,
            "compute_capability": (
                f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability
                else None
            ),
        }


def mem_bandwidth_gbs(memory_clock_khz: int, bus_width_bits: int) -> float:
    """DDR: two transfers per clock. clock(kHz) * 1e3 * width(bytes) * 2 / 1e9."""
    return 2.0 * memory_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9


def fp32_tflops(sm_count: int, clock_khz: int, major: int, minor: int) -> float:
    """One FMA per core per clock = 2 FLOPs."""
    cores = fp32_cores_per_sm(major, minor)
    return 2.0 * sm_count * cores * clock_khz * 1e3 / 1e12


def derive(attrs: dict[str, int]) -> Peaks:
    """Compute peaks from a query_all() attribute dict, tolerating gaps."""
    bw = None
    if "memory_clock_rate_khz" in attrs and "global_memory_bus_width_bits" in attrs:
        bw = mem_bandwidth_gbs(
            attrs["memory_clock_rate_khz"], attrs["global_memory_bus_width_bits"]
        )

    cc = None
    flops = None
    if "compute_capability_major" in attrs and "compute_capability_minor" in attrs:
        cc = (attrs["compute_capability_major"], attrs["compute_capability_minor"])
        if "multiprocessor_count" in attrs and "clock_rate_khz" in attrs:
            flops = fp32_tflops(
                attrs["multiprocessor_count"], attrs["clock_rate_khz"], cc[0], cc[1]
            )

    return Peaks(mem_bandwidth_gbs=bw, fp32_tflops=flops, compute_capability=cc)
