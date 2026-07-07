import ctypes

import pytest

from kernelmeter import cli, hipdrv, peaks, rocmsmi

HIP_SUCCESS = 0
RSMI_SUCCESS = 0

# An MI300X-shaped fake: 304 CUs, 2100 MHz, HBM3 on an 8192-bit bus.
MI300X_ATTRS = {
    5: 2100000,     # clock_rate (khz)
    19: 4194304,    # l2_cache_size (placeholder)
    23: 9,          # compute_capability_major (gfx9xx)
    56: 1024,       # max_threads_per_block
    59: 8192,       # memory_bus_width
    60: 1300000,    # memory_clock_rate (khz): hbm3 quarter-rate, x4 -> 5325 GB/s
    61: 4,          # compute_capability_minor (gfx942)
    63: 304,        # multiprocessor_count
    87: 64,         # warp_size (wavefront)
    10002: 65536,   # max_shared_memory_per_multiprocessor (amd range)
}


class FakeLibHip:
    def hipGetDeviceCount(self, ptr):
        ptr._obj.value = 1
        return HIP_SUCCESS

    def hipRuntimeGetVersion(self, ptr):
        ptr._obj.value = 60443483
        return HIP_SUCCESS

    def hipDeviceGetName(self, buf, size, ordinal):
        buf.value = b"AMD Instinct MI300X"
        return HIP_SUCCESS

    def hipDeviceTotalMem(self, ptr, ordinal):
        ptr._obj.value = 192 * 1024**3
        return HIP_SUCCESS

    def hipDeviceGetAttribute(self, ptr, attr_id, ordinal):
        if attr_id not in MI300X_ATTRS:
            return 1  # hipErrorInvalidValue
        ptr._obj.value = MI300X_ATTRS[attr_id]
        return HIP_SUCCESS


class FakeLibRsmi:
    def rsmi_init(self, flags):
        return RSMI_SUCCESS

    def rsmi_shut_down(self):
        return RSMI_SUCCESS

    def rsmi_num_monitor_devices(self, ptr):
        ptr._obj.value = 1
        return RSMI_SUCCESS

    def rsmi_dev_name_get(self, index, buf, size):
        buf.value = b"AMD Instinct MI300X"
        return RSMI_SUCCESS

    def rsmi_dev_target_graphics_version_get(self, index, ptr):
        ptr._obj.value = 90402
        return RSMI_SUCCESS

    def rsmi_dev_temp_metric_get(self, index, sensor, metric, ptr):
        ptr._obj.value = 41000  # millidegrees
        return RSMI_SUCCESS

    def rsmi_dev_current_socket_power_get(self, index, ptr):
        ptr._obj.value = 176_000_000  # microwatts
        return RSMI_SUCCESS

    def rsmi_dev_power_cap_get(self, index, sensor, ptr):
        ptr._obj.value = 750_000_000
        return RSMI_SUCCESS

    def rsmi_dev_memory_total_get(self, index, mem_type, ptr):
        ptr._obj.value = 192 * 1024**3
        return RSMI_SUCCESS

    def rsmi_dev_memory_usage_get(self, index, mem_type, ptr):
        ptr._obj.value = 3 * 1024**3
        return RSMI_SUCCESS

    def rsmi_dev_gpu_clk_freq_get(self, index, clk_type, ptr):
        freqs = ptr._obj
        freqs.num_supported = 3
        freqs.current = 2
        freqs.frequency[2] = 2_100_000_000 if clk_type == rocmsmi.RSMI_CLK_TYPE_SYS else 1_300_000_000
        return RSMI_SUCCESS


def _fake_hip():
    return hipdrv.HipDriver(lib=FakeLibHip())


def test_hip_attrs_are_named():
    drv = _fake_hip()
    dev = drv.device(0)
    assert dev.name == "AMD Instinct MI300X"
    attrs = hipdrv.query_all(drv, dev)
    assert attrs["multiprocessor_count"] == 304
    assert attrs["memory_bus_width"] == 8192
    assert attrs["warp_size"] == 64
    assert attrs["max_shared_memory_per_multiprocessor"] == 65536  # amd range


def test_amd_arch_detection():
    assert peaks.amd_arch_from(9, 4, "AMD Instinct MI300X") == "cdna3"
    assert peaks.amd_arch_from(9, 0, "AMD Instinct MI250X") == "cdna2"
    assert peaks.amd_arch_from(9, 0, "AMD Instinct MI100") == "cdna1"
    assert peaks.amd_arch_from(11, 0, "Radeon RX 7900 XTX") == "rdna3"
    assert peaks.amd_arch_from(10, 3, "Radeon RX 6900 XT") == "rdna2"
    assert peaks.amd_arch_from(12, 0, "Radeon RX 9070 XT") == "rdna4"
    assert peaks.amd_arch_from(None, None, "mystery") is None


def test_derive_amd_matches_mi300x_spec():
    drv = _fake_hip()
    attrs = hipdrv.query_all(drv, drv.device(0))
    p = peaks.derive_amd(attrs, "cdna3")
    assert p.mem_bandwidth_gbs == pytest.approx(5324.8, rel=0.001)
    assert p.fp32_tflops == pytest.approx(163.4, rel=0.01)
    assert p.fp16_tensor_tflops == pytest.approx(1307.4, rel=0.01)
    assert p.tf32_tensor_tflops is None


def test_rocmsmi_wrapper_reads_values():
    smi = rocmsmi.RocmSmi(lib=FakeLibRsmi())
    assert smi.device_count() == 1
    assert smi.temperature_c(0) == pytest.approx(41.0)
    assert smi.power_w(0) == pytest.approx(176.0)
    assert smi.power_cap_w(0) == pytest.approx(750.0)
    assert smi.vram_bytes(0) == (192 * 1024**3, 3 * 1024**3)
    assert smi.sys_clock_mhz(0) == pytest.approx(2100.0)
    assert smi.mem_clock_mhz(0) == pytest.approx(1300.0)


@pytest.fixture
def amd_machine(monkeypatch):
    """A machine with no CUDA but a working ROCm stack."""
    from kernelmeter.cudadrv import CudaNotAvailableError

    def no_cuda():
        raise CudaNotAvailableError("no NVIDIA driver")

    monkeypatch.setattr(cli, "Driver", no_cuda)
    monkeypatch.setattr(hipdrv, "load_library", lambda: FakeLibHip())
    monkeypatch.setattr(rocmsmi, "load_library", lambda: FakeLibRsmi())


def test_info_on_amd(amd_machine, capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "AMD Instinct MI300X" in out
    assert "cdna3" in out
    assert "5324.8" in out and "163.4" in out.replace("163.43", "163.4")
    assert "live (rocm-smi)" in out
    assert "multiprocessor_count" in out


def test_info_json_on_amd(amd_machine, capsys):
    import json as _json

    assert cli.main(["info", "--json"]) == 0
    payload = _json.loads(capsys.readouterr().out)
    dev = payload["devices"][0]
    assert dev["arch"] == "cdna3"
    assert dev["smi"]["power_cap_w"] == pytest.approx(750.0)


def test_llm_local_on_amd(amd_machine, capsys):
    assert cli.main(["llm", "70b", "--quant", "q4"]) == 0
    out = capsys.readouterr().out
    assert "AMD Instinct MI300X" in out
    assert "yes" in out  # 40.6 GB fits in 192
    assert "131" in out  # 5300/40.6 decode ceiling
    assert "fp32-accumulate" not in out  # no geforce note on amd


def test_bench_device_peaks_falls_through_to_hip(monkeypatch):
    from kernelmeter import bench
    from kernelmeter.cudadrv import CudaNotAvailableError

    class NoCuda:
        def __init__(self, lib=None):
            raise CudaNotAvailableError("no NVIDIA driver")

    monkeypatch.setattr(bench, "Driver", NoCuda)
    monkeypatch.setattr(hipdrv, "load_library", lambda: FakeLibHip())
    p = bench.device_peaks()
    assert p.mem_bandwidth_gbs == pytest.approx(5324.8, rel=0.001)


def test_frequencies_struct_guard():
    # a bogus current index past num_supported must not be trusted
    class BadClocks(FakeLibRsmi):
        def rsmi_dev_gpu_clk_freq_get(self, index, clk_type, ptr):
            ptr._obj.num_supported = 2
            ptr._obj.current = 40
            return RSMI_SUCCESS

    smi = rocmsmi.RocmSmi(lib=BadClocks())
    assert smi.sys_clock_mhz(0) is None
