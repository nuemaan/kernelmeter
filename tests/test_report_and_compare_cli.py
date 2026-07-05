import json

import pytest

from kernelmeter import cli, htmlreport
from kernelmeter.cudadrv import Driver

from conftest import FakeLibCuda


@pytest.fixture
def patched_driver(monkeypatch):
    monkeypatch.setattr(cli, "Driver", lambda: Driver(lib=FakeLibCuda()))


def test_gpus_lists_database(capsys):
    assert cli.main(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "rtx-4090" in out
    assert "h100-sxm" in out
    assert "Tesla T4" in out


def test_gpus_json(capsys):
    assert cli.main(["gpus", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    t4 = next(e for e in payload if e["id"] == "t4")
    assert t4["theoretical_mem_bandwidth_gb_s"] == pytest.approx(320, rel=0.01)


def test_compare_with_intensity(capsys):
    assert cli.main(["compare", "4090", "h100-sxm", "--ai", "0.33"]) == 0
    out = capsys.readouterr().out
    assert "rtx-4090" in out
    assert "vs rtx-4090" in out       # relative column against the first card
    assert "memory-bound" in out      # 0.33 flop/byte is left of every ridge
    assert "* rtx-4090" in out        # overlay legend


def test_compare_with_cost(capsys):
    assert cli.main([
        "compare", "4090", "h100-sxm", "--ai", "0.33",
        "--cost", "4090=0.44,h100-sxm=2.99",
    ]) == 0
    out = capsys.readouterr().out
    assert "TF per $" in out
    assert "0.44" in out


def test_compare_unknown_gpu(capsys):
    assert cli.main(["compare", "4090", "voodoo2"]) == 1
    assert "unknown gpu" in capsys.readouterr().err


def test_roofline_from_database(capsys):
    assert cli.main(["roofline", "--gpu", "4090", "--ai", "680"]) == 0
    out = capsys.readouterr().out
    assert "GeForce RTX 4090" in out
    assert "compute-bound" in out  # 680 flop/byte is right of the ridge


def test_report_from_database(tmp_path, capsys):
    out_file = tmp_path / "card.html"
    assert cli.main(["report", "--gpu", "t4", "--out", str(out_file)]) == 0
    page = out_file.read_text()
    assert "Tesla T4" in page
    assert "<svg" in page
    assert "320" in page  # bandwidth shows up in a tile
    assert "kernelmeter" in page


def test_report_from_device(patched_driver, tmp_path):
    out_file = tmp_path / "dev.html"
    assert cli.main(["report", "--out", str(out_file)]) == 0
    page = out_file.read_text()
    assert "NVIDIA GeForce RTX 3090" in page
    assert "<svg" in page
    assert "multiprocessor_count" in page


def test_report_without_driver_or_gpu(monkeypatch, capsys):
    from kernelmeter.cudadrv import CudaNotAvailableError

    def boom():
        raise CudaNotAvailableError("no driver")

    monkeypatch.setattr(cli, "Driver", boom)
    assert cli.main(["report"]) == 1
    assert "--gpu" in capsys.readouterr().err


def test_svg_is_well_formed():
    svg = htmlreport.svg_roofline(82.6, 1008.0)
    assert svg.startswith("<svg")
    assert svg.count("<svg") == svg.count("</svg>") == 1
    assert "polyline" in svg
    assert "ridge" in svg
