"""Device facts from NVML, the second data source the driver attribute
enum can't give you.

``kernelmeter info`` reports ``cuDeviceGetAttribute`` values. Tools like
Nsight Compute show more (architecture name, real core count, PCIe link,
memory breakdown) because they pull from their own device database and
from NVML. NVML ships with the driver, so this module adds those facts
without a toolkit -- the same ctypes approach as the rest of kernelmeter.

It does not invent ncu-internal metrics (sass_level, ram_type, ...): those
aren't exposed by either the driver or NVML, so they would have to be
hardcoded per board and would go stale. Everything here is read live.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import nvml as _nvml


@dataclass
class DeviceExtras:
    architecture: str | None
    brand: str | None
    num_gpu_cores: int | None
    memory_total_bytes: int | None
    memory_used_bytes: int | None
    memory_free_bytes: int | None
    pcie_gen_current: int | None
    pcie_gen_max: int | None
    pcie_width_current: int | None
    pcie_width_max: int | None
    ecc_enabled: bool | None
    vbios_version: str | None
    driver_version: str | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def from_nvml(n: "_nvml.Nvml", index: int = 0) -> DeviceExtras:
    """Build the extras for one device from an open NVML handle. Each
    query is individually tolerant: an unsupported field becomes None
    rather than failing the whole gather."""
    h = n.device(index)

    def safe(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    arch_id = safe(n.architecture, h)
    brand_id = safe(n.brand, h)
    mem = safe(n.memory_info, h) or (None, None, None)
    pcie = safe(n.pcie_link, h) or (None, None, None, None)

    return DeviceExtras(
        architecture=_nvml.ARCH_NAMES.get(arch_id) if arch_id is not None else None,
        brand=_nvml.BRAND_NAMES.get(brand_id) if brand_id is not None else None,
        num_gpu_cores=safe(n.num_gpu_cores, h),
        memory_total_bytes=mem[0],
        memory_free_bytes=mem[1],
        memory_used_bytes=mem[2],
        pcie_gen_current=pcie[0],
        pcie_gen_max=pcie[1],
        pcie_width_current=pcie[2],
        pcie_width_max=pcie[3],
        ecc_enabled=safe(n.ecc_enabled, h),
        vbios_version=safe(n.vbios_version, h),
        driver_version=safe(n.driver_version),
    )


def gather(index: int = 0, nvml_obj: "_nvml.Nvml | None" = None) -> DeviceExtras | None:
    """Open NVML (if not given one), read the extras, clean up. Returns
    None when NVML isn't available so callers can skip the section."""
    owns = nvml_obj is None
    try:
        n = nvml_obj if nvml_obj is not None else _nvml.Nvml()
    except Exception:
        return None
    try:
        return from_nvml(n, index)
    except Exception:
        return None
    finally:
        if owns:
            try:
                n.close()
            except Exception:
                pass
