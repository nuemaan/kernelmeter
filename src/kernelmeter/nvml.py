"""Live telemetry through NVML (the library behind nvidia-smi).

libnvidia-ml ships with the driver, so this costs no extra dependency.
The point: theoretical peaks assume the max boost clock, but cards
downclock under load. Sampling the actual SM and memory clocks while a
kernel runs lets the bench report what the ceiling really was during the
measurement, not what the spec sheet promised.
"""

from __future__ import annotations

import ctypes
import statistics
import sys
import threading
from dataclasses import dataclass

NVML_SUCCESS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0


class NvmlError(RuntimeError):
    def __init__(self, func: str, code: int):
        super().__init__(f"{func} failed with NVML code {code}")


class NvmlNotAvailableError(RuntimeError):
    pass


def load_library() -> ctypes.CDLL:
    if sys.platform == "darwin":
        raise NvmlNotAvailableError("NVML is not available on macOS")
    names = ("nvml.dll",) if sys.platform == "win32" else ("libnvidia-ml.so.1", "libnvidia-ml.so")
    for name in names:
        try:
            if sys.platform == "win32":
                return ctypes.WinDLL(name)  # pragma: no cover
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise NvmlNotAvailableError("could not load libnvidia-ml; is the NVIDIA driver installed?")


class Nvml:
    """Minimal wrapper. Like cudadrv.Driver, the lib is injectable so the
    tests can run on machines with no NVIDIA driver."""

    def __init__(self, lib=None):
        self._lib = lib if lib is not None else load_library()
        self._check("nvmlInit_v2", self._lib.nvmlInit_v2())

    def _check(self, func: str, code: int) -> None:
        if code != NVML_SUCCESS:
            raise NvmlError(func, code)

    def close(self) -> None:
        self._lib.nvmlShutdown()

    def device(self, index: int = 0) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            "nvmlDeviceGetHandleByIndex",
            self._lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)),
        )
        return handle

    def _uint_query(self, func_name: str, handle, *args) -> int:
        out = ctypes.c_uint(0)
        fn = getattr(self._lib, func_name)
        self._check(func_name, fn(handle, *args, ctypes.byref(out)))
        return out.value

    def sm_clock_mhz(self, handle) -> int:
        return self._uint_query("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_SM)

    def mem_clock_mhz(self, handle) -> int:
        return self._uint_query("nvmlDeviceGetClockInfo", handle, NVML_CLOCK_MEM)

    def max_sm_clock_mhz(self, handle) -> int:
        return self._uint_query("nvmlDeviceGetMaxClockInfo", handle, NVML_CLOCK_SM)

    def max_mem_clock_mhz(self, handle) -> int:
        return self._uint_query("nvmlDeviceGetMaxClockInfo", handle, NVML_CLOCK_MEM)

    def temperature_c(self, handle) -> int:
        return self._uint_query("nvmlDeviceGetTemperature", handle, NVML_TEMPERATURE_GPU)

    def power_w(self, handle) -> float:
        return self._uint_query("nvmlDeviceGetPowerUsage", handle) / 1000.0

    def power_limit_w(self, handle) -> float:
        return self._uint_query("nvmlDeviceGetEnforcedPowerLimit", handle) / 1000.0


@dataclass
class Telemetry:
    sm_clock_mhz: float
    mem_clock_mhz: float
    max_sm_clock_mhz: int
    max_mem_clock_mhz: int
    temperature_c: int
    power_w: float

    @property
    def sm_clock_fraction(self) -> float:
        return self.sm_clock_mhz / self.max_sm_clock_mhz if self.max_sm_clock_mhz else 1.0

    @property
    def mem_clock_fraction(self) -> float:
        return self.mem_clock_mhz / self.max_mem_clock_mhz if self.max_mem_clock_mhz else 1.0


def summarize_samples(
    sm: list[int], mem: list[int], temp: list[int], power: list[float],
    max_sm: int, max_mem: int,
) -> Telemetry:
    return Telemetry(
        sm_clock_mhz=statistics.fmean(sm),
        mem_clock_mhz=statistics.fmean(mem),
        max_sm_clock_mhz=max_sm,
        max_mem_clock_mhz=max_mem,
        temperature_c=max(temp),
        power_w=statistics.fmean(power),
    )


class Monitor:
    """Samples clocks/temperature/power on a background thread while a
    kernel benchmark runs in the main thread."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.02, nvml: Nvml | None = None):
        self._nvml = nvml if nvml is not None else Nvml()
        self._handle = self._nvml.device(device_index)
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sm: list[int] = []
        self._mem: list[int] = []
        self._temp: list[int] = []
        self._power: list[float] = []

    def _sample(self) -> None:
        self._sm.append(self._nvml.sm_clock_mhz(self._handle))
        self._mem.append(self._nvml.mem_clock_mhz(self._handle))
        self._temp.append(self._nvml.temperature_c(self._handle))
        self._power.append(self._nvml.power_w(self._handle))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self) -> None:
        self._stop.clear()
        self._sample()  # always have at least one sample
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Telemetry:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return summarize_samples(
            self._sm, self._mem, self._temp, self._power,
            self._nvml.max_sm_clock_mhz(self._handle),
            self._nvml.max_mem_clock_mhz(self._handle),
        )

    def close(self) -> None:
        self._nvml.close()
