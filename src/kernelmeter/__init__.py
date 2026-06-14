"""kernelmeter: CUDA device attributes without profiling, and kernel
benchmarks measured against the hardware's speed of light."""

from . import extras, occupancy, roofline
from .bench import REGISTRY, BenchResult, BenchSpec, benchmark, device_peaks, run, run_registry
from .cudadrv import CudaDriverError, CudaNotAvailableError, Driver
from .extras import DeviceExtras
from .occupancy import Occupancy
from .peaks import Peaks

__version__ = "0.4.2"

__all__ = [
    "BenchResult",
    "BenchSpec",
    "CudaDriverError",
    "CudaNotAvailableError",
    "DeviceExtras",
    "Driver",
    "Occupancy",
    "Peaks",
    "REGISTRY",
    "benchmark",
    "device_peaks",
    "extras",
    "occupancy",
    "roofline",
    "run",
    "run_registry",
    "__version__",
]
