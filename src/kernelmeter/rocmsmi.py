"""Telemetry through librocm_smi64, the ROCm twin of nvml.py.

Same injectable-library pattern. Only the handful of calls kernelmeter
needs: clocks, temperature, power, vram, and the target graphics version
that pins down the architecture (gfx942 and friends).
"""

from __future__ import annotations

import ctypes
import sys

RSMI_SUCCESS = 0
RSMI_TEMP_CURRENT = 0
RSMI_TEMP_TYPE_EDGE = 0
RSMI_CLK_TYPE_SYS = 0
RSMI_CLK_TYPE_MEM = 4
RSMI_MEM_TYPE_VRAM = 0


class RsmiError(RuntimeError):
    def __init__(self, func: str, code: int):
        super().__init__(f"{func} failed with rsmi_status_t {code}")


class RsmiNotAvailableError(RuntimeError):
    pass


class _Frequencies(ctypes.Structure):
    # rsmi_frequencies_t. The declared array length moved between ROCm
    # releases (32 -> 33), so this allocates well past either and only
    # trusts indices below num_supported.
    _fields_ = [
        ("has_deep_sleep", ctypes.c_bool),
        ("num_supported", ctypes.c_uint32),
        ("current", ctypes.c_uint32),
        ("frequency", ctypes.c_uint64 * 64),
    ]


def load_library() -> ctypes.CDLL:
    if sys.platform == "darwin":
        raise RsmiNotAvailableError("rocm-smi is not available on macOS")
    names = (
        "librocm_smi64.so", "librocm_smi64.so.7", "librocm_smi64.so.6",
        "librocm_smi64.so.5", "/opt/rocm/lib/librocm_smi64.so",
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RsmiNotAvailableError("could not load librocm_smi64")


class RocmSmi:
    def __init__(self, lib=None):
        self._lib = lib if lib is not None else load_library()
        self._check("rsmi_init", self._lib.rsmi_init(0))

    def _check(self, func: str, code: int) -> None:
        if code != RSMI_SUCCESS:
            raise RsmiError(func, code)

    def close(self) -> None:
        self._lib.rsmi_shut_down()

    def device_count(self) -> int:
        n = ctypes.c_uint32(0)
        self._check(
            "rsmi_num_monitor_devices",
            self._lib.rsmi_num_monitor_devices(ctypes.byref(n)),
        )
        return n.value

    def name(self, index: int) -> str:
        buf = ctypes.create_string_buffer(128)
        self._check("rsmi_dev_name_get", self._lib.rsmi_dev_name_get(index, buf, 128))
        return buf.value.decode("utf-8", errors="replace")

    def target_graphics_version(self, index: int) -> int | None:
        out = ctypes.c_uint64(0)
        fn = getattr(self._lib, "rsmi_dev_target_graphics_version_get", None)
        if fn is None or fn(index, ctypes.byref(out)) != RSMI_SUCCESS:
            return None
        return out.value

    def temperature_c(self, index: int) -> float | None:
        out = ctypes.c_int64(0)
        code = self._lib.rsmi_dev_temp_metric_get(
            index, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, ctypes.byref(out)
        )
        return out.value / 1000.0 if code == RSMI_SUCCESS else None

    def power_w(self, index: int) -> float | None:
        out = ctypes.c_uint64(0)
        # rocm 6 prefers the socket-power call; fall back to the average
        fn = getattr(self._lib, "rsmi_dev_current_socket_power_get", None)
        if fn is not None and fn(index, ctypes.byref(out)) == RSMI_SUCCESS:
            return out.value / 1e6
        fn = getattr(self._lib, "rsmi_dev_power_ave_get", None)
        if fn is not None and fn(index, 0, ctypes.byref(out)) == RSMI_SUCCESS:
            return out.value / 1e6
        return None

    def power_cap_w(self, index: int) -> float | None:
        out = ctypes.c_uint64(0)
        code = self._lib.rsmi_dev_power_cap_get(index, 0, ctypes.byref(out))
        return out.value / 1e6 if code == RSMI_SUCCESS else None

    def vram_bytes(self, index: int) -> tuple[int, int] | None:
        total = ctypes.c_uint64(0)
        used = ctypes.c_uint64(0)
        if self._lib.rsmi_dev_memory_total_get(index, RSMI_MEM_TYPE_VRAM, ctypes.byref(total)) != RSMI_SUCCESS:
            return None
        if self._lib.rsmi_dev_memory_usage_get(index, RSMI_MEM_TYPE_VRAM, ctypes.byref(used)) != RSMI_SUCCESS:
            return None
        return total.value, used.value

    def _clock_mhz(self, index: int, clk_type: int) -> float | None:
        freqs = _Frequencies()
        code = self._lib.rsmi_dev_gpu_clk_freq_get(index, clk_type, ctypes.byref(freqs))
        if code != RSMI_SUCCESS:
            return None
        if freqs.current >= min(freqs.num_supported, 64):
            return None
        return freqs.frequency[freqs.current] / 1e6

    def sys_clock_mhz(self, index: int) -> float | None:
        return self._clock_mhz(index, RSMI_CLK_TYPE_SYS)

    def mem_clock_mhz(self, index: int) -> float | None:
        return self._clock_mhz(index, RSMI_CLK_TYPE_MEM)
