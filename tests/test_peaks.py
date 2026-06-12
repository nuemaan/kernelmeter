import pytest

from kernelmeter import attrs, peaks


def test_rtx3090_bandwidth():
    # 9751 MHz GDDR6X, 384-bit bus -> spec sheet says 936 GB/s
    bw = peaks.mem_bandwidth_gbs(9_751_000, 384)
    assert bw == pytest.approx(936, rel=0.01)


def test_rtx3090_fp32():
    # 82 SMs * 128 cores (CC 8.6) * 1695 MHz * 2 -> spec sheet says 35.6 TFLOPS
    tf = peaks.fp32_tflops(82, 1_695_000, 8, 6)
    assert tf == pytest.approx(35.6, rel=0.01)


def test_a100_bandwidth():
    # A100 40GB: 1215 MHz HBM2e, 5120-bit bus -> 1555 GB/s
    bw = peaks.mem_bandwidth_gbs(1_215_000, 5120)
    assert bw == pytest.approx(1555, rel=0.01)


def test_a100_fp32():
    # A100: 108 SMs * 64 cores (CC 8.0) * 1410 MHz * 2 -> 19.5 TFLOPS
    tf = peaks.fp32_tflops(108, 1_410_000, 8, 0)
    assert tf == pytest.approx(19.5, rel=0.01)


def test_unknown_capability_falls_back_to_closest_older():
    assert peaks.fp32_cores_per_sm(9, 2) == peaks.fp32_cores_per_sm(9, 0)
    assert peaks.fp32_cores_per_sm(13, 0) == 128


def test_derive_from_attr_dict(fake_driver):
    dev = fake_driver.device(0)
    p = peaks.derive(attrs.query_all(fake_driver, dev))
    assert p.compute_capability == (8, 6)
    assert p.mem_bandwidth_gbs == pytest.approx(936, rel=0.01)
    assert p.fp32_tflops == pytest.approx(35.6, rel=0.01)


def test_derive_tolerates_missing_attrs():
    p = peaks.derive({"warp_size": 32})
    assert p.mem_bandwidth_gbs is None
    assert p.fp32_tflops is None
    assert p.compute_capability is None
