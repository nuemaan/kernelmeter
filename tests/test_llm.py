import pytest

from kernelmeter import cli, gpus, llm


def test_parse_params():
    assert llm.parse_params("70b") == 70e9
    assert llm.parse_params("8B") == 8e9
    assert llm.parse_params("1.5b") == 1.5e9
    assert llm.parse_params("900m") == 900e6
    assert llm.parse_params("125000000") == 125e6
    with pytest.raises(ValueError):
        llm.parse_params("seventy")


def test_quant_sizes_ring_true():
    # a 70b q4 gguf is roughly 40 GB on disk; 8b f16 is roughly 16 GB
    assert llm.weight_bytes(70e9, llm.QUANTS["q4"]) / 1e9 == pytest.approx(40.6, abs=0.5)
    assert llm.weight_bytes(8e9, llm.QUANTS["f16"]) / 1e9 == pytest.approx(16.0, abs=0.1)


def test_decode_ceiling_is_bandwidth_over_weights():
    # a100-80gb, 70b q4: 2039 GB/s over 40.6 GB of weights -> ~50 t/s
    est = llm.estimate("a100", 2039.0, 312.0, 80 * 1024**3, 70e9, 0.58)
    assert est.fits is True
    assert est.decode_tps == pytest.approx(50.2, abs=0.5)
    assert est.prefill_tps == pytest.approx(2229, rel=0.01)


def test_vram_gate():
    # 70b q4 does not fit a 24GB card, so no throughput is reported
    est = llm.estimate("4090", 1008.0, 330.0, 24 * 1024**3, 70e9, 0.58)
    assert est.fits is False
    assert est.decode_tps is None
    # but an 8b does
    est = llm.estimate("4090", 1008.0, 330.0, 24 * 1024**3, 8e9, 0.58)
    assert est.fits is True
    assert est.decode_tps == pytest.approx(217, abs=2)


def test_overhead_pushes_borderline_models_out():
    # weights alone fit in 11GB, weights + 2GB overhead do not
    est = llm.estimate("2080ti", 616.0, 108.0, 11 * 1024**3, 17e9, 0.58, overhead_gb=2.0)
    assert est.fits is False
    est = llm.estimate("2080ti", 616.0, 108.0, 11 * 1024**3, 17e9, 0.58, overhead_gb=0.0)
    assert est.fits is True


def test_unknown_vram_skips_fit_check():
    est = llm.estimate("mystery", 1000.0, None, None, 70e9, 0.58)
    assert est.fits is None
    assert est.decode_tps is not None
    assert est.prefill_tps is None  # no compute peak given


def test_all_database_cards_have_vram():
    for spec in gpus.DATABASE:
        assert spec.vram_gb > 0, spec.id


@pytest.mark.parametrize(
    "gpu_id,vram",
    [("rtx-4090", 24), ("h100-sxm", 80), ("a100-40gb", 40), ("rtx-2080-ti", 11),
     ("rtx-3080", 10), ("l40s", 48), ("rtx-5090", 32), ("t4", 16)],
)
def test_vram_matches_spec_sheets(gpu_id, vram):
    assert gpus.resolve(gpu_id).vram_gb == vram


def test_cli_llm_with_database_cards(capsys):
    assert cli.main([
        "llm", "70b", "--quant", "q4",
        "--gpus", "4090", "a100-80gb", "--cost", "a100-80gb=1.19",
    ]) == 0
    out = capsys.readouterr().out
    assert "40.6 GB of weights" in out
    assert "no" in out       # 4090 doesn't fit
    assert "yes" in out      # a100 does
    assert "t/s per $" in out
    assert "roofline ceilings" in out


def test_cli_llm_bad_quant(capsys):
    assert cli.main(["llm", "70b", "--quant", "q9", "--gpus", "4090"]) == 1
    assert "unknown quant" in capsys.readouterr().err


def test_cli_llm_local_device(monkeypatch, capsys):
    from kernelmeter.cudadrv import Driver
    from conftest import FakeLibCuda

    monkeypatch.setattr(cli, "Driver", lambda: Driver(lib=FakeLibCuda()))
    assert cli.main(["llm", "8b"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 3090" in out
    assert "yes" in out  # 8b q4 fits in 24GB
