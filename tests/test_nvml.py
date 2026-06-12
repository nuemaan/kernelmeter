import time

import pytest

from kernelmeter import nvml

NVML_SUCCESS = 0


class FakeNvmlLib:
    """Duck-types the libnvidia-ml entry points the wrapper uses.
    Simulates a card boosting to 1590 MHz but holding 1530 under load."""

    def __init__(self):
        self.shutdown_called = False

    def nvmlInit_v2(self):
        return NVML_SUCCESS

    def nvmlShutdown(self):
        self.shutdown_called = True
        return NVML_SUCCESS

    def nvmlDeviceGetHandleByIndex_v2(self, index, ptr):
        ptr._obj.value = 42
        return NVML_SUCCESS

    def nvmlDeviceGetClockInfo(self, handle, clock_type, ptr):
        ptr._obj.value = 1530 if clock_type == nvml.NVML_CLOCK_SM else 4985
        return NVML_SUCCESS

    def nvmlDeviceGetMaxClockInfo(self, handle, clock_type, ptr):
        ptr._obj.value = 1590 if clock_type == nvml.NVML_CLOCK_SM else 5001
        return NVML_SUCCESS

    def nvmlDeviceGetTemperature(self, handle, sensor, ptr):
        ptr._obj.value = 63
        return NVML_SUCCESS

    def nvmlDeviceGetPowerUsage(self, handle, ptr):
        ptr._obj.value = 45200  # milliwatts
        return NVML_SUCCESS

    def nvmlDeviceGetEnforcedPowerLimit(self, handle, ptr):
        ptr._obj.value = 70000
        return NVML_SUCCESS


def test_wrapper_reads_values():
    n = nvml.Nvml(lib=FakeNvmlLib())
    h = n.device(0)
    assert n.sm_clock_mhz(h) == 1530
    assert n.max_sm_clock_mhz(h) == 1590
    assert n.mem_clock_mhz(h) == 4985
    assert n.temperature_c(h) == 63
    assert n.power_w(h) == pytest.approx(45.2)
    assert n.power_limit_w(h) == pytest.approx(70.0)


def test_error_code_raises():
    class Broken(FakeNvmlLib):
        def nvmlDeviceGetTemperature(self, handle, sensor, ptr):
            return 999

    n = nvml.Nvml(lib=Broken())
    with pytest.raises(nvml.NvmlError):
        n.temperature_c(n.device(0))


def test_summarize_samples():
    t = nvml.summarize_samples(
        sm=[1500, 1560], mem=[4985, 4985], temp=[60, 63], power=[44.0, 46.0],
        max_sm=1590, max_mem=5001,
    )
    assert t.sm_clock_mhz == pytest.approx(1530)
    assert t.temperature_c == 63
    assert t.power_w == pytest.approx(45.0)
    assert t.sm_clock_fraction == pytest.approx(1530 / 1590)
    assert t.mem_clock_fraction == pytest.approx(4985 / 5001)


def test_monitor_collects_while_running():
    lib = FakeNvmlLib()
    mon = nvml.Monitor(nvml=nvml.Nvml(lib=lib), interval_s=0.001)
    mon.start()
    time.sleep(0.02)
    t = mon.stop()
    mon.close()
    assert t.sm_clock_mhz == pytest.approx(1530)
    assert t.max_sm_clock_mhz == 1590
    assert lib.shutdown_called
