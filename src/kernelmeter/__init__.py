"""kernelmeter: CUDA device attributes without profiling, and kernel
benchmarks measured against the hardware's speed of light."""

from .bench import REGISTRY, BenchResult, BenchSpec, benchmark, device_peaks, run, run_registry
from .cudadrv import CudaDriverError, CudaNotAvailableError, Driver
from .peaks import Peaks

__version__ = "0.1.0"

__all__ = [
    "BenchResult",
    "BenchSpec",
    "CudaDriverError",
    "CudaNotAvailableError",
    "Driver",
    "Peaks",
    "REGISTRY",
    "benchmark",
    "device_peaks",
    "run",
    "run_registry",
    "__version__",
]
