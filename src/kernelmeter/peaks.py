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


# Dense tensor-core FLOPs per SM per clock, fp16 inputs with fp16
# accumulate. Derived from published board specs (V100 125 TF, T4 65 TF,
# A100 312 TF, 4090 330 TF, H100 SXM 989 TF, ...), which all divide out
# to clean powers of two per SM per clock. GeForce parts run at half
# this rate when accumulating in fp32.
_FP16_TENSOR_FLOPS_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 1024, (7, 2): 1024, (7, 5): 1024,
    (8, 0): 2048, (8, 6): 1024, (8, 7): 1024, (8, 9): 1024,
    (9, 0): 4096,
    (12, 0): 1024, (12, 1): 1024,
}

# Same idea for tf32 (only exists on Ampere and newer).
_TF32_TENSOR_FLOPS_PER_SM: dict[tuple[int, int], int] = {
    (8, 0): 1024, (8, 6): 256, (8, 7): 256, (8, 9): 256,
    (9, 0): 2048,
    (12, 0): 256, (12, 1): 256,
}


# AMD rates per compute unit per clock, by architecture. fp32 includes
# packed/dual-issue where the hardware has it (CDNA3, RDNA3+); the fp16
# rate is matrix-core throughput on CDNA and RDNA3+, packed vector math
# on RDNA2. All verified against vendor spec sheets in the tests.
_AMD_FP32_OPS_PER_CU: dict[str, int] = {
    "cdna1": 128, "cdna2": 128, "cdna3": 256,
    "rdna2": 128, "rdna3": 256, "rdna4": 256,
}
_AMD_FP16_OPS_PER_CU: dict[str, int] = {
    "cdna1": 1024, "cdna2": 1024, "cdna3": 2048,
    "rdna2": 256, "rdna3": 512, "rdna4": 512,
}


def amd_fp32_tflops(cu_count: int, clock_khz: int, arch: str) -> float | None:
    rate = _AMD_FP32_OPS_PER_CU.get(arch.lower())
    if rate is None:
        return None
    return rate * cu_count * clock_khz * 1e3 / 1e12


def amd_fp16_tflops(cu_count: int, clock_khz: int, arch: str) -> float | None:
    rate = _AMD_FP16_OPS_PER_CU.get(arch.lower())
    if rate is None:
        return None
    return rate * cu_count * clock_khz * 1e3 / 1e12


def amd_arch_from(cc_major: int | None, cc_minor: int | None, name: str) -> str | None:
    """Map HIP's compute capability (the gfx version, e.g. gfx942 -> 9.4)
    plus the marketing name to an architecture key. gfx 9.0 covers both
    CDNA1 and CDNA2, so the name breaks the tie there."""
    lowered = name.lower()
    if cc_major == 9:
        if cc_minor == 4:
            return "cdna3"
        if "mi1" in lowered:
            return "cdna1"
        if "mi2" in lowered:
            return "cdna2"
        return "cdna2"
    if cc_major == 10:
        return "rdna2"
    if cc_major == 11:
        return "rdna3"
    if cc_major == 12:
        return "rdna4"
    return None


# transfers per memory clock, per architecture. The HIP runtime reports
# HBM3's quarter-rate clock on CDNA3 (1.3 GHz for 5.2 Gbps/pin on the
# MI300X, verified on hardware), while HBM2/2e and GDDR6 report the
# usual half-rate clock that the x2 DDR formula expects.
_AMD_MEM_TRANSFERS_PER_CLOCK: dict[str, int] = {"cdna3": 4}


def derive_amd(attrs: dict[str, int], arch: str | None) -> Peaks:
    """AMD twin of derive(): bandwidth from memory clock and bus width,
    compute through the per-architecture rate tables."""
    bw = None
    if "memory_clock_rate" in attrs and "memory_bus_width" in attrs:
        transfers = _AMD_MEM_TRANSFERS_PER_CLOCK.get(arch or "", 2)
        bw = (
            transfers * attrs["memory_clock_rate"] * 1e3
            * (attrs["memory_bus_width"] / 8.0) / 1e9
        )

    fp32 = fp16 = None
    if arch and "multiprocessor_count" in attrs and "clock_rate" in attrs:
        fp32 = amd_fp32_tflops(attrs["multiprocessor_count"], attrs["clock_rate"], arch)
        fp16 = amd_fp16_tflops(attrs["multiprocessor_count"], attrs["clock_rate"], arch)

    return Peaks(
        mem_bandwidth_gbs=bw,
        fp32_tflops=fp32,
        compute_capability=None,
        fp16_tensor_tflops=fp16,
        tf32_tensor_tflops=None,
    )


def _tensor_tflops(table: dict, sm_count: int, clock_khz: int, major: int, minor: int) -> float | None:
    rate = table.get((major, minor))
    if rate is None:
        return None
    return rate * sm_count * clock_khz * 1e3 / 1e12


def fp16_tensor_tflops(sm_count: int, clock_khz: int, major: int, minor: int) -> float | None:
    return _tensor_tflops(_FP16_TENSOR_FLOPS_PER_SM, sm_count, clock_khz, major, minor)


def tf32_tensor_tflops(sm_count: int, clock_khz: int, major: int, minor: int) -> float | None:
    return _tensor_tflops(_TF32_TENSOR_FLOPS_PER_SM, sm_count, clock_khz, major, minor)


@dataclass
class Peaks:
    """Theoretical per-device ceilings derived from driver attributes."""

    mem_bandwidth_gbs: float | None
    fp32_tflops: float | None
    compute_capability: tuple[int, int] | None
    fp16_tensor_tflops: float | None = None
    tf32_tensor_tflops: float | None = None

    def as_dict(self) -> dict:
        return {
            "theoretical_mem_bandwidth_gb_s": self.mem_bandwidth_gbs,
            "theoretical_fp32_tflops": self.fp32_tflops,
            "theoretical_fp16_tensor_tflops": self.fp16_tensor_tflops,
            "theoretical_tf32_tensor_tflops": self.tf32_tensor_tflops,
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
    flops = fp16 = tf32 = None
    if "compute_capability_major" in attrs and "compute_capability_minor" in attrs:
        cc = (attrs["compute_capability_major"], attrs["compute_capability_minor"])
        if "multiprocessor_count" in attrs and "clock_rate_khz" in attrs:
            sm, clk = attrs["multiprocessor_count"], attrs["clock_rate_khz"]
            flops = fp32_tflops(sm, clk, cc[0], cc[1])
            fp16 = fp16_tensor_tflops(sm, clk, cc[0], cc[1])
            tf32 = tf32_tensor_tflops(sm, clk, cc[0], cc[1])

    return Peaks(
        mem_bandwidth_gbs=bw,
        fp32_tflops=flops,
        compute_capability=cc,
        fp16_tensor_tflops=fp16,
        tf32_tensor_tflops=tf32,
    )
