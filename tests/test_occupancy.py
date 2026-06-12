import pytest

from kernelmeter import occupancy


CC86 = occupancy.limits_for_cc(8, 6)
CC75 = occupancy.limits_for_cc(7, 5)


def test_full_occupancy_cc86():
    # 256 threads, 40 regs, 8K smem on an RTX 30xx part: the classic
    # calculator says 6 blocks/SM, 48/48 warps, 100%
    occ = occupancy.compute(block=256, regs_per_thread=40, smem_per_block=8192, limits=CC86)
    assert occ.blocks_per_sm == 6
    assert occ.active_warps == 48
    assert occ.pct == pytest.approx(100.0)


def test_register_limited_cc86():
    # bumping to 64 regs/thread: 2048 regs per warp, 32 warps fit, 4 blocks
    occ = occupancy.compute(block=256, regs_per_thread=64, smem_per_block=8192, limits=CC86)
    assert occ.blocks_per_sm == 4
    assert occ.pct == pytest.approx(66.7, abs=0.1)
    assert occ.limited_by == ["registers"]


def test_smem_limited_cc86():
    # 48K smem per block + 1K reserved: only 2 blocks fit in 100K
    occ = occupancy.compute(block=128, regs_per_thread=32, smem_per_block=48 * 1024, limits=CC86)
    assert occ.blocks_per_sm == 2
    assert occ.limited_by == ["shared memory"]


def test_no_regs_no_smem_cc75():
    occ = occupancy.compute(block=128, regs_per_thread=0, smem_per_block=0, limits=CC75)
    assert occ.blocks_per_sm == 8  # 32 max warps / 4 warps per block
    assert occ.pct == pytest.approx(100.0)


def test_tiny_blocks_hit_block_limit():
    # 32-thread blocks on CC 8.6: 16-block cap reached at 16 warps of 48
    occ = occupancy.compute(block=32, regs_per_thread=0, smem_per_block=0, limits=CC86)
    assert occ.blocks_per_sm == 16
    assert occ.limited_by == ["blocks"]
    assert occ.pct == pytest.approx(33.3, abs=0.1)


def test_unknown_cc_falls_back():
    assert occupancy.limits_for_cc(8, 8) == occupancy.limits_for_cc(8, 7)


def test_too_old_cc_raises():
    with pytest.raises(ValueError):
        occupancy.limits_for_cc(3, 5)


def test_block_size_validation():
    with pytest.raises(ValueError):
        occupancy.compute(block=2048, regs_per_thread=0, smem_per_block=0, limits=CC86)


def test_limits_from_attrs():
    attrs = {
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "max_threads_per_multiprocessor": 1536,
        "max_blocks_per_multiprocessor": 16,
        "max_registers_per_multiprocessor": 65536,
        "max_shared_memory_per_multiprocessor": 102400,
        "reserved_shared_memory_per_block": 1024,
    }
    limits = occupancy.limits_from_attrs(attrs)
    assert limits["max_warps"] == 48
    assert limits["smem_gran"] == 128  # granularity comes from the arch table


def test_sweep_shape():
    sweep = occupancy.block_size_sweep(40, 8192, CC86)
    assert [s for s, _ in sweep] == [64, 128, 192, 256, 384, 512, 768, 1024]
    assert all(0 < p <= 100 for _, p in sweep)
