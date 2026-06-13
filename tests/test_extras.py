from kernelmeter import extras, nvml

from test_nvml import FakeNvmlLib


def _fake_nvml():
    return nvml.Nvml(lib=FakeNvmlLib())


def test_from_nvml_builds_extras():
    ex = extras.from_nvml(_fake_nvml(), 0)
    assert ex.architecture == "Turing"
    assert ex.brand == "Tesla"
    assert ex.num_gpu_cores == 2560
    assert ex.memory_total_bytes == 15843721216
    assert ex.pcie_gen_current == 3
    assert ex.pcie_width_max == 16
    assert ex.ecc_enabled is True
    assert ex.vbios_version == "90.04.38.00.03"
    assert ex.driver_version == "535.104.05"


def test_gather_uses_injected_nvml():
    ex = extras.gather(0, nvml_obj=_fake_nvml())
    assert ex is not None
    assert ex.architecture == "Turing"


def test_gather_returns_none_when_nvml_missing(monkeypatch):
    # simulate a machine with no driver: Nvml() construction raises
    def boom(*_a, **_k):
        raise nvml.NvmlNotAvailableError("no driver")

    monkeypatch.setattr(extras._nvml, "Nvml", boom)
    assert extras.gather(0) is None


def test_individual_field_failure_is_tolerated():
    # a card that doesn't report cores shouldn't sink the whole gather
    class NoCores(FakeNvmlLib):
        def nvmlDeviceGetNumGpuCores(self, handle, ptr):
            return nvml.NVML_ERROR_NOT_SUPPORTED

    ex = extras.from_nvml(nvml.Nvml(lib=NoCores()), 0)
    assert ex.num_gpu_cores is None
    assert ex.architecture == "Turing"  # the rest still came through


def test_as_dict_roundtrips():
    ex = extras.from_nvml(_fake_nvml(), 0)
    d = ex.as_dict()
    assert d["architecture"] == "Turing"
    assert d["num_gpu_cores"] == 2560
    assert set(d) == set(ex.__dict__)
