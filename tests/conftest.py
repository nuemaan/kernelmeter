"""A fake libcuda implemented in Python so every test runs without a GPU."""

import ctypes

import pytest

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1

# A plausible RTX 3090: CC 8.6, 82 SMs, 1695 MHz boost, 9751 MHz GDDR6X
# on a 384-bit bus. Published specs: 35.6 FP32 TFLOPS, 936 GB/s.
RTX3090_ATTRS = {
    1: 1024,        # max_threads_per_block
    10: 32,         # warp_size
    13: 1695000,    # clock_rate_khz
    16: 82,         # multiprocessor_count
    36: 9751000,    # memory_clock_rate_khz
    37: 384,        # global_memory_bus_width_bits
    38: 6291456,    # l2_cache_size
    75: 8,          # compute_capability_major
    76: 6,          # compute_capability_minor
    106: 16,        # max_blocks_per_multiprocessor
    150: 7,         # an attribute newer than our name table
}


class FakeLibCuda:
    """Duck-types the handful of libcuda entry points Driver uses."""

    def __init__(self, attrs=None, name=b"NVIDIA GeForce RTX 3090", total_mem=25438126080):
        self.attrs = RTX3090_ATTRS if attrs is None else attrs
        self.name = name
        self.total_mem = total_mem

    def cuInit(self, flags):
        return CUDA_SUCCESS

    def cuDriverGetVersion(self, ptr):
        ptr._obj.value = 12040
        return CUDA_SUCCESS

    def cuDeviceGetCount(self, ptr):
        ptr._obj.value = 1
        return CUDA_SUCCESS

    def cuDeviceGet(self, ptr, ordinal):
        ptr._obj.value = 0
        return CUDA_SUCCESS

    def cuDeviceGetName(self, buf, size, dev):
        buf.value = self.name
        return CUDA_SUCCESS

    def cuDeviceTotalMem_v2(self, ptr, dev):
        ptr._obj.value = self.total_mem
        return CUDA_SUCCESS

    def cuDeviceGetAttribute(self, ptr, attr_id, dev):
        if attr_id not in self.attrs:
            return CUDA_ERROR_INVALID_VALUE
        ptr._obj.value = self.attrs[attr_id]
        return CUDA_SUCCESS


@pytest.fixture
def fake_driver():
    from kernelmeter.cudadrv import Driver

    return Driver(lib=FakeLibCuda())
