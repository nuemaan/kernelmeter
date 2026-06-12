import pytest

from kernelmeter import attrs, peaks


def test_t4_fp16_tensor():
    # T4: 40 SMs, 1590 MHz, CC 7.5 -> spec sheet says 65 TFLOPS fp16
    tf = peaks.fp16_tensor_tflops(40, 1_590_000, 7, 5)
    assert tf == pytest.approx(65.1, rel=0.01)


def test_a100_fp16_and_tf32():
    # A100: 108 SMs, 1410 MHz -> 312 TFLOPS fp16, 156 TFLOPS tf32
    assert peaks.fp16_tensor_tflops(108, 1_410_000, 8, 0) == pytest.approx(312, rel=0.01)
    assert peaks.tf32_tensor_tflops(108, 1_410_000, 8, 0) == pytest.approx(156, rel=0.01)


def test_h100_fp16_tensor():
    # H100 SXM: 132 SMs, 1830 MHz -> 989 TFLOPS dense fp16
    assert peaks.fp16_tensor_tflops(132, 1_830_000, 9, 0) == pytest.approx(989, rel=0.01)


def test_pre_tensor_core_cards_return_none():
    assert peaks.fp16_tensor_tflops(20, 1_700_000, 6, 1) is None
    assert peaks.tf32_tensor_tflops(40, 1_590_000, 7, 5) is None  # Turing has no tf32


def test_derive_includes_tensor_peaks(fake_driver):
    # fake is a 3090: CC 8.6, 82 SMs, 1695 MHz -> 1024 flops/sm/clk = 142 TF
    dev = fake_driver.device(0)
    p = peaks.derive(attrs.query_all(fake_driver, dev))
    assert p.fp16_tensor_tflops == pytest.approx(142.3, rel=0.01)
    assert p.tf32_tensor_tflops == pytest.approx(35.6, rel=0.01)


def test_sustained_peaks_scaling():
    from kernelmeter import bench
    from kernelmeter.nvml import Telemetry

    base = peaks.Peaks(mem_bandwidth_gbs=320.0, fp32_tflops=8.0, compute_capability=(7, 5))
    t = Telemetry(
        sm_clock_mhz=1431, mem_clock_mhz=4500, max_sm_clock_mhz=1590,
        max_mem_clock_mhz=5000, temperature_c=70, power_w=60.0,
    )
    scaled = bench.sustained_peaks(base, t)
    assert scaled.fp32_tflops == pytest.approx(8.0 * 1431 / 1590)
    assert scaled.mem_bandwidth_gbs == pytest.approx(320.0 * 0.9)
    # with an override the override gets scaled instead
    scaled2 = bench.sustained_peaks(base, t, peak_tflops_override=65.0)
    assert scaled2.fp32_tflops == pytest.approx(65.0 * 1431 / 1590)
