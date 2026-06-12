"""Thin ctypes wrapper around the CUDA driver API (libcuda).

The driver API is used instead of the runtime API (libcudart) on purpose:
libcuda ships with the NVIDIA driver itself, so this works on any machine
that can run CUDA programs -- no CUDA toolkit installation required. Its C
ABI is stable, unlike ``cudaDeviceProp`` whose struct layout changes
between toolkit versions.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_NO_DEVICE = 100


class CudaDriverError(RuntimeError):
    """The CUDA driver returned an unexpected error code."""

    def __init__(self, func: str, code: int):
        self.func = func
        self.code = code
        super().__init__(f"{func} failed with CUresult {code}")


class CudaNotAvailableError(RuntimeError):
    """libcuda could not be loaded or no NVIDIA device is present."""


_LIB_CANDIDATES = {
    "linux": ("libcuda.so.1", "libcuda.so"),
    "win32": ("nvcuda.dll",),
    "darwin": (),  # NVIDIA dropped macOS support after CUDA 10.2
}


def load_library() -> ctypes.CDLL:
    """Locate and load libcuda for the current platform."""
    platform = "win32" if sys.platform == "win32" else sys.platform
    candidates = _LIB_CANDIDATES.get(platform, _LIB_CANDIDATES["linux"])
    if not candidates:
        raise CudaNotAvailableError(
            "CUDA is not supported on this platform (no NVIDIA driver for "
            f"{sys.platform!r}). Run kernelmeter on a machine with an NVIDIA GPU."
        )
    errors = []
    for name in candidates:
        try:
            if sys.platform == "win32":
                return ctypes.WinDLL(name)  # pragma: no cover
            return ctypes.CDLL(name)
        except OSError as exc:  # library missing
            errors.append(f"{name}: {exc}")
    raise CudaNotAvailableError(
        "Could not load the CUDA driver library. Is the NVIDIA driver "
        "installed?\nTried: " + "; ".join(errors)
    )


@dataclass
class DeviceHandle:
    ordinal: int
    handle: int
    name: str
    total_mem_bytes: int


class Driver:
    """Minimal, injectable wrapper over the few entry points we need.

    Pass a fake ``lib`` object in tests; production code calls ``Driver()``
    which loads the real libcuda.
    """

    def __init__(self, lib=None):
        self._lib = lib if lib is not None else load_library()
        self._check("cuInit", self._lib.cuInit(0))

    def _check(self, func: str, code: int) -> None:
        if code == CUDA_ERROR_NO_DEVICE:
            raise CudaNotAvailableError(
                "The NVIDIA driver loaded but reported no CUDA-capable device."
            )
        if code != CUDA_SUCCESS:
            raise CudaDriverError(func, code)

    def driver_version(self) -> tuple[int, int]:
        v = ctypes.c_int(0)
        self._check("cuDriverGetVersion", self._lib.cuDriverGetVersion(ctypes.byref(v)))
        return v.value // 1000, (v.value % 1000) // 10

    def device_count(self) -> int:
        n = ctypes.c_int(0)
        self._check("cuDeviceGetCount", self._lib.cuDeviceGetCount(ctypes.byref(n)))
        return n.value

    def device(self, ordinal: int) -> DeviceHandle:
        dev = ctypes.c_int(0)
        self._check("cuDeviceGet", self._lib.cuDeviceGet(ctypes.byref(dev), ordinal))

        buf = ctypes.create_string_buffer(256)
        self._check("cuDeviceGetName", self._lib.cuDeviceGetName(buf, 256, dev))

        mem = ctypes.c_size_t(0)
        # cuDeviceTotalMem_v2 is the modern 64-bit symbol; fall back for
        # exotic builds that only export the unsuffixed name.
        fn = getattr(self._lib, "cuDeviceTotalMem_v2", None) or self._lib.cuDeviceTotalMem
        self._check("cuDeviceTotalMem", fn(ctypes.byref(mem), dev))

        return DeviceHandle(
            ordinal=ordinal,
            handle=dev.value,
            name=buf.value.decode("utf-8", errors="replace"),
            total_mem_bytes=mem.value,
        )

    def attribute(self, device: DeviceHandle, attr_id: int) -> int | None:
        """Query one device attribute. Returns None when the driver does
        not know the attribute id (older driver than the probe range)."""
        out = ctypes.c_int(0)
        code = self._lib.cuDeviceGetAttribute(ctypes.byref(out), attr_id, device.handle)
        if code == CUDA_ERROR_INVALID_VALUE:
            return None
        self._check("cuDeviceGetAttribute", code)
        return out.value
