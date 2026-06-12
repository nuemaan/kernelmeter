import pytest

from kernelmeter import roofline


# T4-like card: 8.14 TFLOP/s, 320 GB/s
PEAK_TF = 8.14
PEAK_BW = 320.0


def test_ridge_point():
    # 8140 GFLOP/s / 320 GB/s = 25.4 flop/byte
    assert roofline.ridge_point(PEAK_TF, PEAK_BW) == pytest.approx(25.4, rel=0.01)


def test_attainable_below_ridge_is_bandwidth_limited():
    # at 1 flop/byte: 320 GB/s * 1 = 0.32 TFLOP/s
    assert roofline.attainable_tflops(1.0, PEAK_TF, PEAK_BW) == pytest.approx(0.32)


def test_attainable_above_ridge_is_compute_limited():
    assert roofline.attainable_tflops(1000.0, PEAK_TF, PEAK_BW) == PEAK_TF


def test_bound_classification():
    assert roofline.bound(1.0, PEAK_TF, PEAK_BW) == "mem"
    assert roofline.bound(100.0, PEAK_TF, PEAK_BW) == "comp"


def test_intensity():
    assert roofline.intensity(2 * 4096**3, 3 * 4096 * 4096 * 4) == pytest.approx(682.7, rel=0.01)


def test_render_shape_and_roof():
    lines = roofline.render(PEAK_TF, PEAK_BW, ai=2.0, width=58, height=12)
    assert len(lines) == 14  # grid + axis line + tick labels
    plot = [line for line in lines[:12]]
    # the roof must contain the ridge marker and the kernel marker
    assert any("x" in line for line in plot)
    assert any("o" in line for line in plot)
    # the flat compute roof sits on the top row, right side
    assert plot[0].rstrip().endswith("*")


def test_render_marker_moves_with_ai():
    low = roofline.render(PEAK_TF, PEAK_BW, ai=0.25)
    high = roofline.render(PEAK_TF, PEAK_BW, ai=200.0)

    def marker_col(lines):
        for line in lines:
            if "o" in line:
                return line.index("o")
        raise AssertionError("no marker drawn")

    assert marker_col(low) < marker_col(high)
