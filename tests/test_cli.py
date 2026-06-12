import json

import pytest

from kernelmeter import cli
from kernelmeter.cudadrv import Driver

from conftest import FakeLibCuda


@pytest.fixture
def patched_driver(monkeypatch):
    monkeypatch.setattr(cli, "Driver", lambda: Driver(lib=FakeLibCuda()))


def test_info_json(patched_driver, capsys):
    assert cli.main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["driver_version"] == "12.4"
    dev = payload["devices"][0]
    assert dev["name"] == "NVIDIA GeForce RTX 3090"
    assert dev["attributes"]["multiprocessor_count"] == 82
    assert dev["derived"]["theoretical_mem_bandwidth_gb_s"] == pytest.approx(936, rel=0.01)


def test_info_human_readable(patched_driver, capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 3090" in out
    assert "multiprocessor_count" in out
    assert "GB/s" in out


def test_info_without_driver(monkeypatch, capsys):
    from kernelmeter.cudadrv import CudaNotAvailableError

    def boom():
        raise CudaNotAvailableError("no NVIDIA driver")

    monkeypatch.setattr(cli, "Driver", boom)
    assert cli.main(["info"]) == 1
    assert "no NVIDIA driver" in capsys.readouterr().err


def test_bench_rejects_file_with_no_registrations(tmp_path, capsys):
    target = tmp_path / "empty.py"
    target.write_text("x = 1\n")
    assert cli.main(["bench", str(target)]) == 1
    assert "registered no benchmarks" in capsys.readouterr().err
