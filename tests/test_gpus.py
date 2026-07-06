import pytest

from kernelmeter import gpus, roofline


def _peaks(gpu_id):
    return gpus.resolve(gpu_id).peaks()


# Every assertion below is a published spec-sheet number. If an entry in
# the database drifts from what the vendor claims, these fail.

@pytest.mark.parametrize(
    "gpu_id,bw,fp32",
    [
        ("t4", 320, 8.1),
        ("v100-sxm2", 900, 15.7),
        ("a100-40gb", 1555, 19.5),
        ("a100-80gb", 2039, 19.5),
        ("h100-sxm", 3347, 66.9),
        ("h100-pcie", 2003, 51.2),
        ("a10", 600, 31.2),
        ("a40", 696, 37.4),
        ("l4", 300, 30.3),
        ("l40s", 864, 91.6),
        ("rtx-a6000", 768, 38.7),
        ("rtx-6000-ada", 960, 91.1),
        ("rtx-3060", 360, 12.7),
        ("rtx-3070", 448, 20.3),
        ("rtx-3080", 760, 29.8),
        ("rtx-3090", 936, 35.6),
        ("rtx-4080", 716.8, 48.7),
        ("rtx-4090", 1008, 82.6),
        ("rtx-5080", 960, 56.3),
        ("rtx-5090", 1792, 104.8),
        ("v100-pcie", 900, 14.1),
        ("a30", 933, 10.3),
        ("l40", 864, 90.5),
        ("rtx-a4000", 448, 19.2),
        ("rtx-a5000", 768, 27.8),
        ("titan-rtx", 672, 16.3),
        ("rtx-2080-ti", 616, 13.45),
        ("rtx-3080-ti", 912, 34.1),
        ("rtx-3090-ti", 1008, 40.0),
        ("rtx-5070", 672, 30.9),
        ("rtx-5070-ti", 896, 43.9),
    ],
)
def test_derived_peaks_match_spec_sheets(gpu_id, bw, fp32):
    p = _peaks(gpu_id)
    assert p.mem_bandwidth_gbs == pytest.approx(bw, rel=0.01)
    assert p.fp32_tflops == pytest.approx(fp32, rel=0.01)


def test_known_tensor_rates():
    # cards where the datasheet tensor number and the datasheet boost clock
    # are mutually consistent (H100's are not, so it isn't asserted here)
    assert _peaks("t4").fp16_tensor_tflops == pytest.approx(65, rel=0.01)
    assert _peaks("a100-80gb").fp16_tensor_tflops == pytest.approx(312, rel=0.01)
    assert _peaks("rtx-4090").fp16_tensor_tflops == pytest.approx(330, rel=0.01)


def test_resolve_exact_and_fuzzy():
    assert gpus.resolve("rtx-4090").id == "rtx-4090"
    assert gpus.resolve("4090").id == "rtx-4090"
    assert gpus.resolve("T4").id == "t4"
    assert gpus.resolve("l40s").id == "l40s"


def test_resolve_prefers_exact_over_ti_variants():
    # bare numbers keep resolving to the base card, not the ti
    assert gpus.resolve("3090").id == "rtx-3090"
    assert gpus.resolve("3080").id == "rtx-3080"
    assert gpus.resolve("5070").id == "rtx-5070"
    assert gpus.resolve("3090-ti").id == "rtx-3090-ti"
    assert gpus.resolve("a5000").id == "rtx-a5000"


def test_resolve_ambiguous_lists_candidates():
    with pytest.raises(gpus.UnknownGpuError, match="ambiguous"):
        gpus.resolve("a100")
    with pytest.raises(gpus.UnknownGpuError, match="h100-sxm"):
        gpus.resolve("h100")
    with pytest.raises(gpus.UnknownGpuError, match="ambiguous"):
        gpus.resolve("v100")


def test_resolve_unknown():
    with pytest.raises(gpus.UnknownGpuError, match="unknown gpu"):
        gpus.resolve("voodoo2")


def test_multi_roofline_render():
    roofs = [
        ("rtx-4090", 82.6, 1008.0),
        ("h100-sxm", 66.9, 3347.0),
    ]
    lines = roofline.render_multi(roofs, ai=0.33)
    text = "\n".join(lines)
    assert "* rtx-4090" in text
    assert "o h100-sxm" in text
    assert "|" in text  # the ai marker column
    assert "flop/byte" in text


def test_multi_roofline_limits():
    with pytest.raises(ValueError):
        roofline.render_multi([])
    with pytest.raises(ValueError):
        roofline.render_multi([("x", 1.0, 100.0)] * 6)
