import pytest

from kernelmeter import cli
from kernelmeter.cudadrv import Driver

from conftest import FakeLibCuda


@pytest.fixture
def patched_driver(monkeypatch):
    monkeypatch.setattr(cli, "Driver", lambda: Driver(lib=FakeLibCuda()))


def test_roofline_from_device(patched_driver, capsys):
    assert cli.main(["roofline", "--ai", "2.0"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 3090" in out
    assert "ridge point" in out
    assert "memory-bound" in out
    assert "*" in out  # the chart got drawn


def test_roofline_manual_peaks_need_no_device(monkeypatch, capsys):
    from kernelmeter.cudadrv import CudaNotAvailableError

    def boom():
        raise CudaNotAvailableError("no driver")

    monkeypatch.setattr(cli, "Driver", boom)
    assert cli.main(["roofline", "--peak-bw", "320", "--peak-tflops", "8.1"]) == 0
    assert "ridge point" in capsys.readouterr().out


def test_roofline_no_device_no_peaks_errors(monkeypatch, capsys):
    from kernelmeter.cudadrv import CudaNotAvailableError

    def boom():
        raise CudaNotAvailableError("no driver")

    monkeypatch.setattr(cli, "Driver", boom)
    assert cli.main(["roofline"]) == 1
    assert "--peak-bw" in capsys.readouterr().err


def test_occupancy_with_cc(capsys):
    assert cli.main(["occupancy", "--block", "256", "--regs", "40", "--smem", "8192",
                     "--cc", "8.6"]) == 0
    out = capsys.readouterr().out
    assert "100.0% (48/48 warps per SM)" in out
    assert "block size" in out


def test_occupancy_from_device(patched_driver, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_device_attrs", lambda ordinal=0: {
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "max_threads_per_multiprocessor": 1536,
        "max_blocks_per_multiprocessor": 16,
        "max_registers_per_multiprocessor": 65536,
        "max_shared_memory_per_multiprocessor": 102400,
        "reserved_shared_memory_per_block": 1024,
    })
    assert cli.main(["occupancy", "--block", "256", "--regs", "64"]) == 0
    out = capsys.readouterr().out
    assert "registers" in out


def test_occupancy_bad_cc(capsys):
    assert cli.main(["occupancy", "--block", "256", "--cc", "3.5"]) == 1
    assert "error" in capsys.readouterr().err
