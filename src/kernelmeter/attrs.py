"""Enumerate every device attribute the driver knows about.

Nsight Compute's ``device__attribute_*`` values come straight from
``cuDeviceGetAttribute``, so we ask the driver directly. Every id from 1
to PROBE_MAX is probed: ids the driver rejects are skipped, ids that
succeed but aren't in the name table yet are reported as ``attribute_<id>``.
"""

from __future__ import annotations

from .cudadrv import DeviceHandle, Driver

PROBE_MAX = 192

# CU_DEVICE_ATTRIBUTE_* names (snake_cased), keyed by enum value.
# Deprecated/reserved slots (44, 92-94, ...) are intentionally absent.
KNOWN_ATTRS: dict[int, str] = {
    1: "max_threads_per_block",
    2: "max_block_dim_x",
    3: "max_block_dim_y",
    4: "max_block_dim_z",
    5: "max_grid_dim_x",
    6: "max_grid_dim_y",
    7: "max_grid_dim_z",
    8: "max_shared_memory_per_block",
    9: "total_constant_memory",
    10: "warp_size",
    11: "max_pitch",
    12: "max_registers_per_block",
    13: "clock_rate_khz",
    14: "texture_alignment",
    15: "gpu_overlap",
    16: "multiprocessor_count",
    17: "kernel_exec_timeout",
    18: "integrated",
    19: "can_map_host_memory",
    20: "compute_mode",
    21: "maximum_texture1d_width",
    22: "maximum_texture2d_width",
    23: "maximum_texture2d_height",
    24: "maximum_texture3d_width",
    25: "maximum_texture3d_height",
    26: "maximum_texture3d_depth",
    27: "maximum_texture2d_layered_width",
    28: "maximum_texture2d_layered_height",
    29: "maximum_texture2d_layered_layers",
    30: "surface_alignment",
    31: "concurrent_kernels",
    32: "ecc_enabled",
    33: "pci_bus_id",
    34: "pci_device_id",
    35: "tcc_driver",
    36: "memory_clock_rate_khz",
    37: "global_memory_bus_width_bits",
    38: "l2_cache_size",
    39: "max_threads_per_multiprocessor",
    40: "async_engine_count",
    41: "unified_addressing",
    42: "maximum_texture1d_layered_width",
    43: "maximum_texture1d_layered_layers",
    45: "maximum_texture2d_gather_width",
    46: "maximum_texture2d_gather_height",
    47: "maximum_texture3d_width_alternate",
    48: "maximum_texture3d_height_alternate",
    49: "maximum_texture3d_depth_alternate",
    50: "pci_domain_id",
    51: "texture_pitch_alignment",
    52: "maximum_texturecubemap_width",
    53: "maximum_texturecubemap_layered_width",
    54: "maximum_texturecubemap_layered_layers",
    55: "maximum_surface1d_width",
    56: "maximum_surface2d_width",
    57: "maximum_surface2d_height",
    58: "maximum_surface3d_width",
    59: "maximum_surface3d_height",
    60: "maximum_surface3d_depth",
    61: "maximum_surface1d_layered_width",
    62: "maximum_surface1d_layered_layers",
    63: "maximum_surface2d_layered_width",
    64: "maximum_surface2d_layered_height",
    65: "maximum_surface2d_layered_layers",
    66: "maximum_surfacecubemap_width",
    67: "maximum_surfacecubemap_layered_width",
    68: "maximum_surfacecubemap_layered_layers",
    69: "maximum_texture1d_linear_width",
    70: "maximum_texture2d_linear_width",
    71: "maximum_texture2d_linear_height",
    72: "maximum_texture2d_linear_pitch",
    73: "maximum_texture2d_mipmapped_width",
    74: "maximum_texture2d_mipmapped_height",
    75: "compute_capability_major",
    76: "compute_capability_minor",
    77: "maximum_texture1d_mipmapped_width",
    78: "stream_priorities_supported",
    79: "global_l1_cache_supported",
    80: "local_l1_cache_supported",
    81: "max_shared_memory_per_multiprocessor",
    82: "max_registers_per_multiprocessor",
    83: "managed_memory",
    84: "multi_gpu_board",
    85: "multi_gpu_board_group_id",
    86: "host_native_atomic_supported",
    87: "single_to_double_precision_perf_ratio",
    88: "pageable_memory_access",
    89: "concurrent_managed_access",
    90: "compute_preemption_supported",
    91: "can_use_host_pointer_for_registered_mem",
    95: "cooperative_launch",
    96: "cooperative_multi_device_launch",
    97: "max_shared_memory_per_block_optin",
    98: "can_flush_remote_writes",
    99: "host_register_supported",
    100: "pageable_memory_access_uses_host_page_tables",
    101: "direct_managed_mem_access_from_host",
    102: "virtual_memory_management_supported",
    103: "handle_type_posix_file_descriptor_supported",
    104: "handle_type_win32_handle_supported",
    105: "handle_type_win32_kmt_handle_supported",
    106: "max_blocks_per_multiprocessor",
    107: "generic_compression_supported",
    108: "max_persisting_l2_cache_size",
    109: "max_access_policy_window_size",
    110: "gpu_direct_rdma_with_cuda_vmm_supported",
    111: "reserved_shared_memory_per_block",
    112: "sparse_cuda_array_supported",
    113: "read_only_host_register_supported",
    114: "timeline_semaphore_interop_supported",
    115: "memory_pools_supported",
    116: "gpu_direct_rdma_supported",
    117: "gpu_direct_rdma_flush_writes_options",
    118: "gpu_direct_rdma_writes_ordering",
    119: "mempool_supported_handle_types",
    120: "cluster_launch",
    121: "deferred_mapping_cuda_array_supported",
    122: "can_use_64_bit_stream_mem_ops",
    123: "can_use_stream_wait_value_nor",
    124: "dma_buf_supported",
    125: "ipc_event_supported",
    126: "mem_sync_domain_count",
    127: "tensor_map_access_supported",
    128: "handle_type_fabric_supported",
    129: "unified_function_pointers",
    130: "numa_config",
    131: "numa_id",
    132: "multicast_supported",
    133: "mps_enabled",
    134: "host_numa_id",
    135: "d3d12_cig_supported",
    136: "mem_decompress_algorithm_mask",
    137: "mem_decompress_maximum_length",
    138: "vulkan_cig_supported",
    139: "gpu_pci_device_id",
    140: "gpu_pci_subsystem_id",
    141: "host_numa_virtual_memory_management_supported",
    142: "host_numa_memory_pools_supported",
    143: "host_numa_multinode_ipc_supported",
    144: "host_memory_pools_supported",
    145: "host_virtual_memory_management_supported",
    146: "host_alloc_dma_buf_supported",
    147: "only_partial_host_native_atomic_supported",
    148: "atomic_reduction_supported",
    # 149 is CU_DEVICE_ATTRIBUTE_MAX, a sentinel rather than a real
    # attribute, so it stops here. Anything the driver adds beyond this is
    # still reported generically as attribute_<id> by the probe below.
}


def attr_name(attr_id: int) -> str:
    return KNOWN_ATTRS.get(attr_id, f"attribute_{attr_id}")


def query_all(driver: Driver, device: DeviceHandle, probe_max: int = PROBE_MAX) -> dict[str, int]:
    """Return {attribute_name: value} for every attribute this driver supports."""
    out: dict[str, int] = {}
    for attr_id in range(1, probe_max + 1):
        value = driver.attribute(device, attr_id)
        if value is not None:
            out[attr_name(attr_id)] = value
    return out
