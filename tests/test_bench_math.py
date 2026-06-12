import pytest

from kernelmeter import bench


def test_summarize_times():
    mean, median, fastest = bench.summarize_times([2.0, 1.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert median == pytest.approx(2.0)
    assert fastest == pytest.approx(1.0)


def test_achieved_gbps():
    # 1 GB moved in 1 ms = 1000 GB/s
    assert bench.achieved_gbps(10**9, 1.0) == pytest.approx(1000.0)


def test_achieved_tflops():
    # 1e12 FLOPs in 1000 ms = 1 TFLOP/s
    assert bench.achieved_tflops(10**12, 1000.0) == pytest.approx(1.0)


def test_pct_of_peak():
    assert bench.pct_of_peak(450.0, 900.0) == pytest.approx(50.0)
    assert bench.pct_of_peak(450.0, None) is None


def test_resolve_metric():
    assert bench._resolve(None, ()) is None
    assert bench._resolve(42, ()) == 42
    assert bench._resolve(lambda x: x * 2, (21,)) == 42


def test_benchmark_decorator_registers():
    before = len(bench.REGISTRY)

    @bench.benchmark("toy", args=tuple, bytes_per_call=8)
    def toy():
        pass

    try:
        assert len(bench.REGISTRY) == before + 1
        spec = bench.REGISTRY[-1]
        assert spec.name == "toy"
        assert spec.fn is toy
    finally:
        bench.REGISTRY.pop()
