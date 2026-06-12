"""Theoretical occupancy from block size, registers and shared memory.

This reimplements the model behind NVIDIA's old occupancy calculator
spreadsheet. It is a model, not a measurement: the real limiter can also
be launch bounds or the driver, but for the everyday question ("why is my
occupancy 50% and what do I change?") the model is what you want.

Registers are allocated per warp in units of 256. Shared memory is
allocated per block in architecture-specific units, and on Ampere and
newer the runtime reserves about 1 KiB per block on top of what you ask
for.
"""

from __future__ import annotations

from dataclasses import dataclass

REG_ALLOC_UNIT = 256  # registers, rounded up per warp

# Per-SM limits by compute capability. Values from the CUDA C programming
# guide's compute-capability table.
ARCH_LIMITS: dict[tuple[int, int], dict] = {
    (7, 0): dict(max_warps=64, max_blocks=32, regs=65536, smem=98304, smem_gran=256, reserved_smem=0),
    (7, 5): dict(max_warps=32, max_blocks=16, regs=65536, smem=65536, smem_gran=256, reserved_smem=0),
    (8, 0): dict(max_warps=64, max_blocks=32, regs=65536, smem=167936, smem_gran=128, reserved_smem=1024),
    (8, 6): dict(max_warps=48, max_blocks=16, regs=65536, smem=102400, smem_gran=128, reserved_smem=1024),
    (8, 7): dict(max_warps=48, max_blocks=16, regs=65536, smem=167936, smem_gran=128, reserved_smem=1024),
    (8, 9): dict(max_warps=48, max_blocks=24, regs=65536, smem=102400, smem_gran=128, reserved_smem=1024),
    (9, 0): dict(max_warps=64, max_blocks=32, regs=65536, smem=233472, smem_gran=128, reserved_smem=1024),
    (12, 0): dict(max_warps=48, max_blocks=24, regs=65536, smem=102400, smem_gran=128, reserved_smem=1024),
}


def limits_for_cc(major: int, minor: int) -> dict:
    if (major, minor) in ARCH_LIMITS:
        return ARCH_LIMITS[(major, minor)]
    older = [cc for cc in ARCH_LIMITS if cc <= (major, minor)]
    if not older:
        raise ValueError(f"no occupancy data for compute capability {major}.{minor}")
    return ARCH_LIMITS[max(older)]


def limits_from_attrs(attrs: dict[str, int]) -> dict:
    """Build the limits dict from live device attributes, falling back to
    the arch table for allocation granularities the driver doesn't report."""
    cc = (attrs["compute_capability_major"], attrs["compute_capability_minor"])
    arch = limits_for_cc(*cc)
    return dict(
        max_warps=attrs.get("max_threads_per_multiprocessor", arch["max_warps"] * 32) // 32,
        max_blocks=attrs.get("max_blocks_per_multiprocessor", arch["max_blocks"]),
        regs=attrs.get("max_registers_per_multiprocessor", arch["regs"]),
        smem=attrs.get("max_shared_memory_per_multiprocessor", arch["smem"]),
        smem_gran=arch["smem_gran"],
        reserved_smem=attrs.get("reserved_shared_memory_per_block", arch["reserved_smem"]),
    )


def _ceil_to(value: int, unit: int) -> int:
    return (value + unit - 1) // unit * unit


@dataclass
class Occupancy:
    blocks_per_sm: int
    active_warps: int
    max_warps: int
    pct: float
    limited_by: list[str]

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compute(block: int, regs_per_thread: int, smem_per_block: int, limits: dict) -> Occupancy:
    if block < 1 or block > 1024:
        raise ValueError("block size must be 1..1024")
    warps_per_block = (block + 31) // 32

    by = {"blocks": limits["max_blocks"], "threads": limits["max_warps"] // warps_per_block}
    if regs_per_thread:
        regs_per_warp = _ceil_to(regs_per_thread * 32, REG_ALLOC_UNIT)
        by["registers"] = (limits["regs"] // regs_per_warp) // warps_per_block
    smem_total = smem_per_block + limits["reserved_smem"] if smem_per_block else 0
    if smem_total:
        by["shared memory"] = limits["smem"] // _ceil_to(smem_total, limits["smem_gran"])

    blocks = min(by.values())
    warps = blocks * warps_per_block
    return Occupancy(
        blocks_per_sm=blocks,
        active_warps=warps,
        max_warps=limits["max_warps"],
        pct=100.0 * warps / limits["max_warps"],
        limited_by=[k for k, v in by.items() if v == blocks],
    )


def block_size_sweep(
    regs_per_thread: int, smem_per_block: int, limits: dict,
    sizes: tuple = (64, 128, 192, 256, 384, 512, 768, 1024),
) -> list[tuple[int, float]]:
    return [(s, compute(s, regs_per_thread, smem_per_block, limits).pct) for s in sizes]
