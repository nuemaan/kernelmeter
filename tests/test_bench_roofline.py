import pytest

from kernelmeter import bench
from kernelmeter.peaks import Peaks

T4 = Peaks(mem_bandwidth_gbs=320.0, fp32_tflops=8.14, compute_capability=(7, 5))


def test_memory_bound_kernel_scored_against_bandwidth_roof():
    # vector add: 0.33 flop/byte, achieved 240 GB/s and 0.08 TFLOP/s
    ai, bound, pct = bench.roofline_score(
        nbytes=12 * 10**8, nflops=4 * 10**8, gbps=240.0, tflops=0.08, peaks=T4
    )
    assert ai == pytest.approx(1 / 3)
    assert bound == "mem"
    # attainable at 1/3 flop/byte = 320/3 GFLOP/s = 0.1067 TF; 0.08/0.1067 = 75%
    assert pct == pytest.approx(75.0, abs=0.1)


def test_compute_bound_kernel_scored_against_compute_roof():
    ai, bound, pct = bench.roofline_score(
        nbytes=10**6, nflops=10**9, gbps=100.0, tflops=4.07, peaks=T4
    )
    assert ai == pytest.approx(1000.0)
    assert bound == "comp"
    assert pct == pytest.approx(50.0)


def test_bytes_only_falls_back_to_bandwidth_pct():
    ai, bound, pct = bench.roofline_score(
        nbytes=10**9, nflops=None, gbps=160.0, tflops=None, peaks=T4
    )
    assert ai is None
    assert bound == "mem"
    assert pct == pytest.approx(50.0)


def test_flops_only_falls_back_to_compute_pct():
    ai, bound, pct = bench.roofline_score(
        nbytes=None, nflops=10**9, gbps=None, tflops=4.07, peaks=T4
    )
    assert bound == "comp"
    assert pct == pytest.approx(50.0)


def test_peak_override_replaces_fp32_roof():
    # tensor-core kernel: judge against 65 TFLOP/s, not the fp32 peak
    _, _, pct = bench.roofline_score(
        nbytes=None, nflops=10**9, gbps=None, tflops=32.5, peaks=T4, peak_tflops_override=65.0
    )
    assert pct == pytest.approx(50.0)


def test_no_metrics_no_score():
    assert bench.roofline_score(None, None, None, None, T4) == (None, None, None)


def test_diff_results_flags_regressions():
    baseline = [
        {"name": "fast", "ms_median": 1.0},
        {"name": "slow", "ms_median": 1.0},
        {"name": "gone", "ms_median": 1.0},
    ]
    results = [
        bench.BenchResult(name="fast", ms_mean=0, ms_median=1.01, ms_min=0),
        bench.BenchResult(name="slow", ms_mean=0, ms_median=1.5, ms_min=0),
        bench.BenchResult(name="new", ms_mean=0, ms_median=9.9, ms_min=0),
    ]
    rows, regressions = bench.diff_results(baseline, results)
    assert len(rows) == 2  # "gone" and "new" don't match up
    assert regressions == ["slow"]
