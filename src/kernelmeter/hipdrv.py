"""Thin ctypes wrapper around the HIP runtime (libamdhip64).

The ROCm twin of cudadrv.py: same injectable-library pattern, same
role. libamdhip64 ships with the ROCm driver stack, and the C entry
points used here are stable across ROCm 5/6.

The attribute name table is generated from hipDeviceAttribute_t in
AMD's hip_runtime_api.h, the same way the CUDA table tracks the driver
enum. Ids the runtime rejects are skipped; ids newer than the table are
reported as attribute_<id>.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

HIP_SUCCESS = 0


class HipError(RuntimeError):
    def __init__(self, func: str, code: int):
        super().__init__(f"{func} failed with hipError_t {code}")


class HipNotAvailableError(RuntimeError):
    pass


def load_library() -> ctypes.CDLL:
    if sys.platform == "darwin":
        raise HipNotAvailableError("ROCm is not available on macOS")
    names = (
        "libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5",
        "/opt/rocm/lib/libamdhip64.so",
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise HipNotAvailableError(
        "could not load libamdhip64; is the ROCm driver stack installed?"
    )


@dataclass
class HipDeviceHandle:
    ordinal: int
    name: str
    total_mem_bytes: int


# hipDeviceAttribute_t, generated from hip_runtime_api.h. The cuda-style
# block sits below 200; amd-specific attributes start at 10000.
KNOWN_ATTRS: dict[int, str] = {
    0: "ecc_enabled",
    1: "access_policy_max_window_size",
    2: "async_engine_count",
    3: "can_map_host_memory",
    4: "can_use_host_pointer_for_registered_mem",
    5: "clock_rate",
    6: "compute_mode",
    7: "compute_preemption_supported",
    8: "concurrent_kernels",
    9: "concurrent_managed_access",
    10: "cooperative_launch",
    11: "cooperative_multi_device_launch",
    12: "device_overlap",
    13: "direct_managed_mem_access_from_host",
    14: "global_l1_cache_supported",
    15: "host_native_atomic_supported",
    16: "integrated",
    17: "is_multi_gpu_board",
    18: "kernel_exec_timeout",
    19: "l2_cache_size",
    20: "local_l1_cache_supported",
    21: "luid",
    22: "luid_device_node_mask",
    23: "compute_capability_major",
    24: "managed_memory",
    25: "max_blocks_per_multi_processor",
    26: "max_block_dim_x",
    27: "max_block_dim_y",
    28: "max_block_dim_z",
    29: "max_grid_dim_x",
    30: "max_grid_dim_y",
    31: "max_grid_dim_z",
    32: "max_surface1_d",
    33: "max_surface1_d_layered",
    34: "max_surface2_d",
    35: "max_surface2_d_layered",
    36: "max_surface3_d",
    37: "max_surface_cubemap",
    38: "max_surface_cubemap_layered",
    39: "max_texture1_d_width",
    40: "max_texture1_d_layered",
    41: "max_texture1_d_linear",
    42: "max_texture1_d_mipmap",
    43: "max_texture2_d_width",
    44: "max_texture2_d_height",
    45: "max_texture2_d_gather",
    46: "max_texture2_d_layered",
    47: "max_texture2_d_linear",
    48: "max_texture2_d_mipmap",
    49: "max_texture3_d_width",
    50: "max_texture3_d_height",
    51: "max_texture3_d_depth",
    52: "max_texture3_d_alt",
    53: "max_texture_cubemap",
    54: "max_texture_cubemap_layered",
    55: "max_threads_dim",
    56: "max_threads_per_block",
    57: "max_threads_per_multi_processor",
    58: "max_pitch",
    59: "memory_bus_width",
    60: "memory_clock_rate",
    61: "compute_capability_minor",
    62: "multi_gpu_board_group_id",
    63: "multiprocessor_count",
    64: "unused1",
    65: "pageable_memory_access",
    66: "pageable_memory_access_uses_host_page_tables",
    67: "pci_bus_id",
    68: "pci_device_id",
    69: "pci_domain_id",
    70: "persisting_l2_cache_max_size",
    71: "max_registers_per_block",
    72: "max_registers_per_multiprocessor",
    73: "reserved_shared_mem_per_block",
    74: "max_shared_memory_per_block",
    75: "shared_mem_per_block_optin",
    76: "shared_mem_per_multiprocessor",
    77: "single_to_double_precision_perf_ratio",
    78: "stream_priorities_supported",
    79: "surface_alignment",
    80: "tcc_driver",
    81: "texture_alignment",
    82: "texture_pitch_alignment",
    83: "total_constant_memory",
    84: "total_global_mem",
    85: "unified_addressing",
    86: "unused2",
    87: "warp_size",
    88: "memory_pools_supported",
    89: "virtual_memory_management_supported",
    90: "host_register_supported",
    91: "memory_pool_supported_handle_types",
    10000: "clock_instruction_rate",
    10001: "unused3",
    10002: "max_shared_memory_per_multiprocessor",
    10003: "unused4",
    10004: "unused5",
    10005: "hdp_mem_flush_cntl",
    10006: "hdp_reg_flush_cntl",
    10007: "cooperative_multi_device_unmatched_func",
    10008: "cooperative_multi_device_unmatched_grid_dim",
    10009: "cooperative_multi_device_unmatched_block_dim",
    10010: "cooperative_multi_device_unmatched_shared_mem",
    10011: "is_large_bar",
    10012: "asic_revision",
    10013: "can_use_stream_wait_value",
    10014: "image_support",
    10015: "physical_multi_processor_count",
    10016: "fine_grain_support",
    10017: "wall_clock_rate",
    10018: "number_of_xccs",
    10019: "max_available_vgprs_per_thread",
    10020: "pci_chip_id",
}

CUDA_STYLE_MAX = 200
AMD_SPECIFIC_RANGE = (10000, 10050)


def attr_name(attr_id: int) -> str:
    return KNOWN_ATTRS.get(attr_id, f"attribute_{attr_id}")


class HipDriver:
    """Mirrors cudadrv.Driver closely enough that peaks/info code can
    treat the two interchangeably."""

    def __init__(self, lib=None):
        self._lib = lib if lib is not None else load_library()
        count = ctypes.c_int(0)
        code = self._lib.hipGetDeviceCount(ctypes.byref(count))
        if code != HIP_SUCCESS or count.value < 1:
            raise HipNotAvailableError(
                "the HIP runtime loaded but reported no usable device"
            )
        self._count = count.value

    def _check(self, func: str, code: int) -> None:
        if code != HIP_SUCCESS:
            raise HipError(func, code)

    def runtime_version(self) -> int:
        v = ctypes.c_int(0)
        self._check("hipRuntimeGetVersion", self._lib.hipRuntimeGetVersion(ctypes.byref(v)))
        return v.value

    def device_count(self) -> int:
        return self._count

    def device(self, ordinal: int) -> HipDeviceHandle:
        buf = ctypes.create_string_buffer(256)
        self._check("hipDeviceGetName", self._lib.hipDeviceGetName(buf, 256, ordinal))
        mem = ctypes.c_size_t(0)
        self._check("hipDeviceTotalMem", self._lib.hipDeviceTotalMem(ctypes.byref(mem), ordinal))
        return HipDeviceHandle(
            ordinal=ordinal,
            name=buf.value.decode("utf-8", errors="replace"),
            total_mem_bytes=mem.value,
        )

    def attribute(self, device: HipDeviceHandle, attr_id: int) -> int | None:
        out = ctypes.c_int(0)
        code = self._lib.hipDeviceGetAttribute(ctypes.byref(out), attr_id, device.ordinal)
        if code != HIP_SUCCESS:
            return None  # unknown/unsupported id on this runtime
        return out.value


def query_all(driver: HipDriver, device: HipDeviceHandle) -> dict[str, int]:
    """Every attribute this runtime supports, named where known."""
    out: dict[str, int] = {}
    for attr_id in range(0, CUDA_STYLE_MAX):
        value = driver.attribute(device, attr_id)
        if value is not None:
            out[attr_name(attr_id)] = value
    for attr_id in range(*AMD_SPECIFIC_RANGE):
        value = driver.attribute(device, attr_id)
        if value is not None:
            out[attr_name(attr_id)] = value
    return out
