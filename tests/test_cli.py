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


@pytest.fixture
def patched_nvml(monkeypatch):
    from kernelmeter import extras, nvml

    from test_nvml import FakeNvmlLib

    real_nvml = nvml.Nvml  # capture before patching to avoid self-recursion
    monkeypatch.setattr(
        extras._nvml, "Nvml", lambda *a, **k: real_nvml(lib=FakeNvmlLib())
    )


def test_info_json_includes_nvml(patched_driver, patched_nvml, capsys):
    assert cli.main(["info", "--json"]) == 0
    dev = json.loads(capsys.readouterr().out)["devices"][0]
    assert dev["nvml"]["architecture"] == "Turing"
    assert dev["nvml"]["num_gpu_cores"] == 2560
    assert dev["nvml"]["pcie_gen_max"] == 3


def test_info_human_shows_nvml(patched_driver, patched_nvml, capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Turing" in out
    assert "2560 CUDA cores" in out
    assert "pcie link" in out


def test_info_json_nvml_null_without_nvml(patched_driver, monkeypatch, capsys):
    from kernelmeter import extras, nvml

    def boom(*_a, **_k):
        raise nvml.NvmlNotAvailableError("no driver")

    monkeypatch.setattr(extras._nvml, "Nvml", boom)
    assert cli.main(["info", "--json"]) == 0
    dev = json.loads(capsys.readouterr().out)["devices"][0]
    assert dev["nvml"] is None


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
