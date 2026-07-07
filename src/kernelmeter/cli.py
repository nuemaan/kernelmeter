"""kernelmeter command line interface.

  kernelmeter info                  dump every device attribute + derived peaks
  kernelmeter bench file.py         run all @kernelmeter.benchmark specs in file
  kernelmeter roofline              draw the device roofline, place your kernel on it
  kernelmeter occupancy             theoretical occupancy from block/regs/smem
  kernelmeter ceiling               measure real achievable bandwidth and FP32
  kernelmeter gpus                  list the built-in card database
  kernelmeter compare 4090 h100-sxm compare cards at your kernel's intensity
  kernelmeter llm 70b --gpus 4090   token/s ceilings for llm inference
  kernelmeter report                write a shareable single-file html report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from . import attrs as _attrs
from . import gpus as _gpus
from . import occupancy as _occupancy
from . import peaks as _peaks
from . import roofline as _roofline
from .cudadrv import CudaNotAvailableError, Driver


def _fmt(value: float | None, suffix: str = "", nd: int = 1) -> str:
    return f"{value:.{nd}f}{suffix}" if value is not None else "-"


def _device_attrs(ordinal: int = 0) -> dict[str, int]:
    driver = Driver()
    return _attrs.query_all(driver, driver.device(ordinal))


def _print_live_telemetry(ordinal: int) -> None:
    """Current clocks/temp/power via NVML; quietly skipped when absent."""
    try:
        from . import nvml as _nvml

        n = _nvml.Nvml()
    except Exception:
        return
    try:
        h = n.device(ordinal)
        print(
            f"  live: sm {n.sm_clock_mhz(h)}/{n.max_sm_clock_mhz(h)} MHz, "
            f"mem {n.mem_clock_mhz(h)}/{n.max_mem_clock_mhz(h)} MHz, "
            f"{n.temperature_c(h)}C, "
            f"{n.power_w(h):.1f}/{n.power_limit_w(h):.0f}W"
        )
    except Exception:
        pass
    finally:
        n.close()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def gather_info(driver: Driver) -> dict:
    from . import extras as _extras

    major, minor = driver.driver_version()
    devices = []
    for ordinal in range(driver.device_count()):
        dev = driver.device(ordinal)
        attributes = _attrs.query_all(driver, dev)
        peaks = _peaks.derive(attributes)
        nvml_extras = _extras.gather(ordinal)
        devices.append(
            {
                "ordinal": ordinal,
                "name": dev.name,
                "total_memory_bytes": dev.total_mem_bytes,
                "derived": peaks.as_dict(),
                "nvml": nvml_extras.as_dict() if nvml_extras else None,
                "attributes": attributes,
            }
        )
    return {"driver_version": f"{major}.{minor}", "devices": devices}


def _print_nvml_extras(nvml: dict) -> None:
    """Print the NVML-sourced facts, skipping fields the card didn't report."""
    arch = nvml.get("architecture")
    cores = nvml.get("num_gpu_cores")
    if arch or cores:
        bits = []
        if arch:
            bits.append(arch)
        if cores:
            bits.append(f"{cores} CUDA cores")
        print("  architecture (nvml)       : " + ", ".join(bits))
    gen, gen_max = nvml.get("pcie_gen_current"), nvml.get("pcie_gen_max")
    w, w_max = nvml.get("pcie_width_current"), nvml.get("pcie_width_max")
    if gen and w:
        print(f"  pcie link (nvml)          : gen{gen}/{gen_max} x{w}/{w_max}")
    total, used = nvml.get("memory_total_bytes"), nvml.get("memory_used_bytes")
    if total:
        print(
            f"  memory in use (nvml)      : {used / 2**20:.0f} / "
            f"{total / 2**20:.0f} MiB"
        )
    if nvml.get("ecc_enabled") is not None:
        print(f"  ecc (nvml)                : {'on' if nvml['ecc_enabled'] else 'off'}")
    if nvml.get("vbios_version"):
        print(f"  vbios (nvml)              : {nvml['vbios_version']}")


def gather_info_amd() -> dict:
    """AMD twin of gather_info, via the HIP runtime plus rocm-smi."""
    from . import hipdrv as _hip
    from . import rocmsmi as _rsmi

    driver = _hip.HipDriver()
    smi = None
    try:
        smi = _rsmi.RocmSmi()
    except Exception:
        smi = None

    devices = []
    for ordinal in range(driver.device_count()):
        dev = driver.device(ordinal)
        attributes = _hip.query_all(driver, dev)
        arch = _peaks.amd_arch_from(
            attributes.get("compute_capability_major"),
            attributes.get("compute_capability_minor"),
            dev.name,
        )
        peaks = _peaks.derive_amd(attributes, arch)
        smi_facts = None
        if smi is not None:
            try:
                vram = smi.vram_bytes(ordinal)
                smi_facts = {
                    "temperature_c": smi.temperature_c(ordinal),
                    "power_w": smi.power_w(ordinal),
                    "power_cap_w": smi.power_cap_w(ordinal),
                    "sys_clock_mhz": smi.sys_clock_mhz(ordinal),
                    "mem_clock_mhz": smi.mem_clock_mhz(ordinal),
                    "vram_total_bytes": vram[0] if vram else None,
                    "vram_used_bytes": vram[1] if vram else None,
                }
            except Exception:
                smi_facts = None
        devices.append(
            {
                "ordinal": ordinal,
                "name": dev.name,
                "total_memory_bytes": dev.total_mem_bytes,
                "arch": arch,
                "derived": peaks.as_dict(),
                "smi": smi_facts,
                "attributes": attributes,
            }
        )
    if smi is not None:
        smi.close()
    return {"hip_runtime_version": driver.runtime_version(), "devices": devices}


def _cmd_info_amd(args: argparse.Namespace) -> int:
    info = gather_info_amd()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"HIP runtime version : {info['hip_runtime_version']}")
    for dev in info["devices"]:
        gib = dev["total_memory_bytes"] / 2**30
        print(f"\nDevice {dev['ordinal']}: {dev['name']} ({gib:.1f} GiB)")
        derived = dev["derived"]
        print(f"  architecture              : {dev['arch'] or '-'}")
        print(
            "  theoretical mem bandwidth : "
            + _fmt(derived["theoretical_mem_bandwidth_gb_s"], " GB/s")
        )
        print(
            "  theoretical FP32 peak     : "
            + _fmt(derived["theoretical_fp32_tflops"], " TFLOP/s", nd=2)
        )
        if derived.get("theoretical_fp16_tensor_tflops"):
            print(
                "  theoretical fp16 matrix   : "
                + _fmt(derived["theoretical_fp16_tensor_tflops"], " TFLOP/s (dense)", nd=2)
            )
        smi = dev.get("smi")
        if smi:
            bits = []
            if smi.get("sys_clock_mhz"):
                bits.append(f"sclk {smi['sys_clock_mhz']:.0f} MHz")
            if smi.get("mem_clock_mhz"):
                bits.append(f"mclk {smi['mem_clock_mhz']:.0f} MHz")
            if smi.get("temperature_c") is not None:
                bits.append(f"{smi['temperature_c']:.0f}C")
            if smi.get("power_w") is not None and smi.get("power_cap_w"):
                bits.append(f"{smi['power_w']:.0f}/{smi['power_cap_w']:.0f}W")
            if smi.get("vram_used_bytes") is not None and smi.get("vram_total_bytes"):
                bits.append(
                    f"vram {smi['vram_used_bytes'] / 2**20:.0f}/"
                    f"{smi['vram_total_bytes'] / 2**20:.0f} MiB"
                )
            if bits:
                print("  live (rocm-smi): " + ", ".join(bits))
        print(f"\n  {'attribute':<48} value")
        print(f"  {'-' * 48} {'-' * 12}")
        for name, value in dev["attributes"].items():
            print(f"  {name:<48} {value}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        driver = Driver()
    except CudaNotAvailableError as exc:
        from .hipdrv import HipNotAvailableError

        try:
            return _cmd_info_amd(args)
        except HipNotAvailableError:
            pass
        except Exception as amd_exc:
            print(f"error: hip backend failed: {amd_exc}", file=sys.stderr)
            return 1
        print(f"error: {exc}", file=sys.stderr)
        return 1

    info = gather_info(driver)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"CUDA driver version : {info['driver_version']}")
    for dev in info["devices"]:
        gib = dev["total_memory_bytes"] / 2**30
        print(f"\nDevice {dev['ordinal']}: {dev['name']} ({gib:.1f} GiB)")
        derived = dev["derived"]
        print(f"  compute capability        : {derived['compute_capability']}")
        print(
            "  theoretical mem bandwidth : "
            + _fmt(derived["theoretical_mem_bandwidth_gb_s"], " GB/s")
        )
        print(
            "  theoretical FP32 peak     : "
            + _fmt(derived["theoretical_fp32_tflops"], " TFLOP/s", nd=2)
        )
        if derived.get("theoretical_fp16_tensor_tflops"):
            print(
                "  theoretical fp16 tensor   : "
                + _fmt(derived["theoretical_fp16_tensor_tflops"], " TFLOP/s (dense)", nd=2)
            )
        if derived.get("theoretical_tf32_tensor_tflops"):
            print(
                "  theoretical tf32 tensor   : "
                + _fmt(derived["theoretical_tf32_tensor_tflops"], " TFLOP/s (dense)", nd=2)
            )
        if dev.get("nvml"):
            _print_nvml_extras(dev["nvml"])
        _print_live_telemetry(dev["ordinal"])
        print(f"\n  {'attribute':<48} value")
        print(f"  {'-' * 48} {'-' * 12}")
        for name, value in dev["attributes"].items():
            print(f"  {name:<48} {value}")
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def _load_module(path: str) -> None:
    spec = importlib.util.spec_from_file_location("kernelmeter_bench_target", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench as _bench

    _load_module(args.file)
    if not _bench.REGISTRY:
        print(
            f"error: {args.file} registered no benchmarks. "
            "Decorate your kernels with @kernelmeter.benchmark(...).",
            file=sys.stderr,
        )
        return 1

    results = _bench.run_registry(flush_l2=not args.no_flush_l2, device_index=args.device)

    if args.save:
        with open(args.save, "w") as fh:
            json.dump([r.as_dict() for r in results], fh, indent=2)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        header = (
            f"{'kernel':<24} {'median ms':>10} {'GB/s':>9} {'TFLOP/s':>9} "
            f"{'bound':>6} {'%roof':>7} {'vs ref':>8} {'correct':>8}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            if r.error:
                print(f"{r.name:<24} failed: {r.error}")
                continue
            correct = "-" if r.correct is None else ("PASS" if r.correct else "FAIL")
            speedup = f"{r.speedup_vs_ref:.2f}x" if r.speedup_vs_ref else "-"
            print(
                f"{r.name:<24} {r.ms_median:>10.4f} {_fmt(r.gbps):>9} "
                f"{_fmt(r.tflops, nd=2):>9} {r.bound or '-':>6} "
                f"{_fmt(r.pct_roofline, '%'):>7} {speedup:>8} {correct:>8}"
            )

        if any(r.sm_clock_mhz for r in results):
            print()
            theader = (
                f"{'telemetry':<24} {'sm MHz':>10} {'mem MHz':>9} "
                f"{'temp':>6} {'power':>7} {'%roof@clk':>10}"
            )
            print(theader)
            print("-" * len(theader))
            for r in results:
                if not r.sm_clock_mhz:
                    continue
                print(
                    f"{r.name:<24} {r.sm_clock_mhz:>5.0f}/{r.max_sm_clock_mhz:<4} "
                    f"{_fmt(r.mem_clock_mhz, nd=0):>9} {r.temperature_c or '-':>5}C "
                    f"{_fmt(r.power_w, 'W'):>7} {_fmt(r.pct_roof_sustained, '%'):>10}"
                )
            print(
                "%roof@clk scores against the ceiling at the clocks the card "
                "actually held during the run"
            )

    ok = all(r.error is None and r.correct in (None, True) for r in results)

    if args.compare:
        with open(args.compare) as fh:
            baseline = json.load(fh)
        rows, regressions = _bench.diff_results(baseline, results)
        if rows:
            print(f"\n{'kernel':<24} {'baseline ms':>12} {'now ms':>10} {'delta':>8}")
            for name, old_ms, new_ms, delta in rows:
                print(f"{name:<24} {old_ms:>12.4f} {new_ms:>10.4f} {delta:>+7.1f}%")
        for name in regressions:
            print(f"regression: {name} is more than 5% slower than the baseline")
        if regressions:
            ok = False

    return 0 if ok else 1


# ---------------------------------------------------------------------------
# roofline
# ---------------------------------------------------------------------------

def cmd_roofline(args: argparse.Namespace) -> int:
    peak_bw, peak_tf = args.peak_bw, args.peak_tflops
    name = None
    if args.gpu:
        try:
            spec = _gpus.resolve(args.gpu)
        except _gpus.UnknownGpuError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        name = spec.name
        peaks = spec.peaks()
        peak_bw = peak_bw or peaks.mem_bandwidth_gbs
        peak_tf = peak_tf or (
            peaks.fp16_tensor_tflops if args.tensor else peaks.fp32_tflops
        )
    elif peak_bw is None or peak_tf is None:
        try:
            driver = Driver()
            dev = driver.device(args.device)
            name = dev.name
            peaks = _peaks.derive(_attrs.query_all(driver, dev))
            peak_bw = peak_bw or peaks.mem_bandwidth_gbs
            if args.tensor:
                if peaks.fp16_tensor_tflops is None:
                    print(
                        "error: no tensor-core rate known for this card; "
                        "pass --peak-tflops instead.",
                        file=sys.stderr,
                    )
                    return 1
                peak_tf = peak_tf or peaks.fp16_tensor_tflops
            else:
                peak_tf = peak_tf or peaks.fp32_tflops
        except CudaNotAvailableError:
            pass
    if not peak_bw or not peak_tf:
        print(
            "error: no CUDA device found. Pass --gpu (e.g. --gpu 4090) or "
            "--peak-bw/--peak-tflops to draw a roofline for any card.",
            file=sys.stderr,
        )
        return 1

    ridge = _roofline.ridge_point(peak_tf, peak_bw)
    if name:
        print(name if args.gpu else f"Device {args.device}: {name}")
    roof_kind = "fp16 tensor" if args.tensor else "fp32"
    print(f"  peak bandwidth : {peak_bw:.1f} GB/s")
    print(f"  peak compute   : {peak_tf:.2f} TFLOP/s ({roof_kind})")
    print(f"  ridge point    : {ridge:.1f} flop/byte\n")
    for line in _roofline.render(peak_tf, peak_bw, ai=args.ai):
        print(line)
    if args.ai is not None:
        attainable = _roofline.attainable_tflops(args.ai, peak_tf, peak_bw)
        which = _roofline.bound(args.ai, peak_tf, peak_bw)
        kind = "memory" if which == "mem" else "compute"
        print(
            f"\nat {args.ai:g} flop/byte the kernel is {kind}-bound; "
            f"attainable: {attainable:.2f} TFLOP/s"
        )
    return 0


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------

def cmd_occupancy(args: argparse.Namespace) -> int:
    if args.cc:
        try:
            major, minor = (int(x) for x in args.cc.split("."))
            limits = _occupancy.limits_for_cc(major, minor)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        source = f"compute capability {args.cc}"
        if (major, minor) not in _occupancy.ARCH_LIMITS:
            known = max(cc for cc in _occupancy.ARCH_LIMITS if cc <= (major, minor))
            source += f" (limits borrowed from {known[0]}.{known[1]}, nearest known arch)"
    else:
        try:
            limits = _occupancy.limits_from_attrs(_device_attrs(args.device))
            source = f"device {args.device}"
        except CudaNotAvailableError:
            print(
                "error: no CUDA device found. Pass --cc (e.g. --cc 8.6) to "
                "model a card you don't have.",
                file=sys.stderr,
            )
            return 1

    try:
        occ = _occupancy.compute(args.block, args.regs, args.smem, limits)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"occupancy for {source}")
    print(f"  block={args.block} regs/thread={args.regs} smem/block={args.smem}\n")
    print(f"  occupancy    : {occ.pct:.1f}% ({occ.active_warps}/{occ.max_warps} warps per SM)")
    print(f"  blocks per SM: {occ.blocks_per_sm}")
    print(f"  limited by   : {', '.join(occ.limited_by)}\n")

    sweep = _occupancy.block_size_sweep(args.regs, args.smem, limits)
    print("  block size " + "".join(f"{s:>7}" for s, _ in sweep))
    print("  occupancy  " + "".join(f"{p:>6.0f}%" for _, p in sweep))
    return 0


# ---------------------------------------------------------------------------
# gpus / compare
# ---------------------------------------------------------------------------

def cmd_gpus(args: argparse.Namespace) -> int:
    if args.json:
        out = []
        for spec in _gpus.DATABASE:
            entry = {
                "id": spec.id, "name": spec.name, "vendor": spec.vendor,
                "arch": spec.arch_label, "tdp_w": spec.tdp_w,
                "vram_gb": spec.vram_gb,
            }
            entry.update(spec.peaks().as_dict())
            out.append(entry)
        print(json.dumps(out, indent=2))
        return 0
    header = (
        f"{'id':<14} {'name':<20} {'arch':>6} {'SM/CU':>5} {'vram':>5} "
        f"{'bw GB/s':>8} {'fp32 TF':>8} {'fp16 TC':>8} {'tdp':>5}"
    )
    print(header)
    print("-" * len(header))
    for spec in _gpus.DATABASE:
        p = spec.peaks()
        fp16 = f"{p.fp16_tensor_tflops:.0f}" if p.fp16_tensor_tflops else "-"
        print(
            f"{spec.id:<14} {spec.name:<20} {spec.arch_label:>6} "
            f"{spec.sm_count:>5} {spec.vram_gb:>3}GB {p.mem_bandwidth_gbs:>8.0f} "
            f"{p.fp32_tflops:>8.1f} {fp16:>8} {spec.tdp_w:>4}W"
        )
    return 0


def _parse_costs(text: str) -> dict[str, float]:
    """'4090=0.44,h100-sxm=2.99' -> {resolved_id: dollars_per_hour}"""
    costs = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        costs[_gpus.resolve(key).id] = float(value)
    return costs


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        specs = [_gpus.resolve(q) for q in args.gpu]
        costs = _parse_costs(args.cost) if args.cost else {}
    except (_gpus.UnknownGpuError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ai = args.ai
    rows = []
    for spec in specs:
        p = spec.peaks()
        tf = (p.fp16_tensor_tflops or p.fp32_tflops) if args.tensor else p.fp32_tflops
        attainable = _roofline.attainable_tflops(ai, tf, p.mem_bandwidth_gbs) if ai else None
        rows.append((spec, p, tf, attainable))

    base = rows[0][3]  # first card's attainable, for the relative column
    header = f"{'card':<14} {'bw GB/s':>8} {'fp32 TF':>8} {'fp16 TC':>8} {'ridge':>6}"
    if ai:
        header += f" {'@' + format(ai, 'g') + ' TF':>9} {'vs ' + rows[0][0].id:>12}"
    if costs:
        header += f" {'$/hr':>6} {'TF per $':>9}"
    print(header)
    print("-" * len(header))
    for spec, p, tf, attainable in rows:
        fp16 = f"{p.fp16_tensor_tflops:.0f}" if p.fp16_tensor_tflops else "-"
        line = (
            f"{spec.id:<14} {p.mem_bandwidth_gbs:>8.0f} {p.fp32_tflops:>8.1f} "
            f"{fp16:>8} {_roofline.ridge_point(tf, p.mem_bandwidth_gbs):>6.1f}"
        )
        if ai:
            rel = f"{attainable / base:.2f}x" if base else "-"
            line += f" {attainable:>9.2f} {rel:>12}"
        if costs:
            cost = costs.get(spec.id)
            if cost and attainable:
                line += f" {cost:>6.2f} {attainable / cost:>9.2f}"
            elif cost:
                line += f" {cost:>6.2f} {'-':>9}"
            else:
                line += f" {'-':>6} {'-':>9}"
        print(line)

    if ai:
        kind = "memory" if all(
            _roofline.bound(ai, tf, p.mem_bandwidth_gbs) == "mem" for _, p, tf, _ in rows
        ) else "mixed"
        if kind == "memory":
            print("\nat this intensity every card is memory-bound: bandwidth is what you're buying")

    if len(rows) <= len(_roofline.MULTI_SYMBOLS):
        print()
        roofs = [(spec.id, tf, p.mem_bandwidth_gbs) for spec, p, tf, _ in rows]
        for line in _roofline.render_multi(roofs, ai=ai):
            print(line)
    return 0


# ---------------------------------------------------------------------------
# llm
# ---------------------------------------------------------------------------

def _local_device_peaks(index: int):
    """(name, Peaks, total_mem_bytes, tdp_w) for the local device, trying
    CUDA first and the HIP runtime second. None when neither is present."""
    try:
        driver = Driver()
        dev = driver.device(index)
        p = _peaks.derive(_attrs.query_all(driver, dev))
        tdp = None
        try:
            from . import nvml as _nvml

            nv = _nvml.Nvml()
            tdp = nv.power_limit_w(nv.device(index))
            nv.close()
        except Exception:
            pass
        return dev.name, p, dev.total_mem_bytes, tdp
    except CudaNotAvailableError:
        pass
    try:
        from . import hipdrv as _hip

        driver = _hip.HipDriver()
        dev = driver.device(index)
        attributes = _hip.query_all(driver, dev)
        arch = _peaks.amd_arch_from(
            attributes.get("compute_capability_major"),
            attributes.get("compute_capability_minor"),
            dev.name,
        )
        p = _peaks.derive_amd(attributes, arch)
        tdp = None
        try:
            from . import rocmsmi as _rsmi

            smi = _rsmi.RocmSmi()
            tdp = smi.power_cap_w(index)
            smi.close()
        except Exception:
            pass
        return dev.name, p, dev.total_mem_bytes, tdp
    except Exception:
        return None


def cmd_llm(args: argparse.Namespace) -> int:
    from . import llm as _llm

    try:
        params = _llm.parse_params(args.params)
        active = _llm.parse_params(args.active_params) if args.active_params else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.bytes_per_param:
        bpp, quant_label = args.bytes_per_param, f"{args.bytes_per_param} bytes/param"
    else:
        if args.quant not in _llm.QUANTS:
            print(
                f"error: unknown quant {args.quant!r} "
                f"(choices: {', '.join(_llm.QUANTS)})",
                file=sys.stderr,
            )
            return 1
        bpp = _llm.QUANTS[args.quant]
        quant_label = f"{args.quant} (~{bpp} bytes/param)"

    if args.num_gpus < 1 or args.batch < 1:
        print("error: --num-gpus and --batch must be at least 1", file=sys.stderr)
        return 1

    n = args.num_gpus
    # (display name, bandwidth, compute peak, vram bytes, vram display,
    #  $/hr per card, tdp watts, consumer?)
    cards = []
    if args.gpus:
        try:
            queries = [q for chunk in args.gpus for q in chunk.split(",") if q]
            specs = [_gpus.resolve(q) for q in queries]
            costs = _parse_costs(args.cost) if args.cost else {}
        except (_gpus.UnknownGpuError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for spec in specs:
            p = spec.peaks()
            compute = p.fp16_tensor_tflops or p.fp32_tflops
            if spec.is_consumer and p.fp16_tensor_tflops:
                compute /= 2  # fp32-accumulate rate, what inference stacks use
            vram_label = f"{n}x{spec.vram_gb}GB" if n > 1 else f"{spec.vram_gb}GB"
            cards.append(
                (spec.id, p.mem_bandwidth_gbs, compute, spec.vram_gb * 1024**3,
                 vram_label, costs.get(spec.id), spec.tdp_w, spec.is_consumer)
            )
    else:
        local = _local_device_peaks(args.device)
        if local is None:
            print(
                "error: no CUDA or ROCm device found. Pass --gpus (e.g. --gpus "
                "4090 a100-80gb) to estimate for cards from the database.",
                file=sys.stderr,
            )
            return 1
        name, p, total_mem, tdp = local
        compute = p.fp16_tensor_tflops or p.fp32_tflops
        consumer = any(m in name.lower() for m in ("geforce", "titan"))
        if consumer and p.fp16_tensor_tflops:
            compute /= 2
        cards.append(
            (name, p.mem_bandwidth_gbs, compute, total_mem,
             f"{total_mem / 2**30:.0f}GB", None, tdp, consumer)
        )

    weights_gb = _llm.weight_bytes(params, bpp) / 1e9
    line = f"{args.params} model at {quant_label}: {weights_gb:.1f} GB of weights"
    if active:
        line += f", {args.active_params} active ({_llm.weight_bytes(active, bpp) / 1e9:.1f} GB read per token)"
    if n > 1:
        line += f", split over {n} gpus"
    print(line + "\n")

    any_cost = any(c[5] for c in cards)
    batched = args.batch > 1
    header = f"{'card':<24} {'vram':>8} {'fits':>5} {'decode t/s':>11}"
    if batched:
        header += f" {'/stream':>8}"
    header += f" {'prefill t/s':>12}"
    if any_cost:
        header += f" {'$/hr':>6} {'t/s per $':>10}"
    if args.per_watt:
        header += f" {'t/s per W':>10}"
    print(header)
    print("-" * len(header))
    for name, bw, compute, vram_bytes, vram_label, cost, tdp, _consumer in cards:
        est = _llm.estimate(
            name, bw, compute, vram_bytes, params, bpp,
            overhead_gb=args.overhead_gb, active_params=active,
            num_gpus=n, batch=args.batch,
        )
        fits = "-" if est.fits is None else ("yes" if est.fits else "no")
        decode = f"{est.decode_tps:.0f}" if est.decode_tps else "-"
        prefill = f"{est.prefill_tps:.0f}" if est.prefill_tps else "-"
        row = f"{name:<24} {vram_label:>8} {fits:>5} {decode:>11}"
        if batched:
            stream = f"{est.per_stream_tps:.0f}" if est.per_stream_tps else "-"
            row += f" {stream:>8}"
        row += f" {prefill:>12}"
        if any_cost:
            total_cost = cost * n if cost else None
            if total_cost and est.decode_tps:
                row += f" {total_cost:>6.2f} {est.decode_tps / total_cost:>10.1f}"
            else:
                row += f" {'-':>6} {'-':>10}"
        if args.per_watt:
            if tdp and est.decode_tps:
                row += f" {est.decode_tps / (tdp * n):>10.2f}"
            else:
                row += f" {'-':>10}"
        print(row)

    notes = [
        "these are roofline ceilings, not predictions: well-tuned stacks land at "
        "50-85%\nof them, none land above. kv cache and activations need room on "
        f"top of the\nweights ({args.overhead_gb:g} GB per gpu assumed here)."
    ]
    if any(c[7] for c in cards):
        notes.append(
            "geforce/titan prefill uses the fp32-accumulate tensor rate "
            "(half the fp16 peak)."
        )
    if n > 1:
        notes.append(
            "multi-gpu numbers assume an ideal tensor-parallel split; "
            "interconnect costs extra."
        )
    print("\n" + "\n".join(notes))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    from . import __version__
    from . import htmlreport as _html

    if args.gpu:
        try:
            spec = _gpus.resolve(args.gpu)
        except _gpus.UnknownGpuError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        page = _html.build(
            name=spec.name,
            derived=spec.peaks().as_dict(),
            version=__version__,
            subtitle=(
                f"{spec.sm_count} {'CUs' if spec.vendor == 'amd' else 'SMs'}, "
                f"{spec.tdp_w} W, from the spec database"
            ),
        )
    else:
        try:
            driver = Driver()
        except CudaNotAvailableError as exc:
            print(
                f"error: {exc}\nUse --gpu (e.g. --gpu 4090) to build a report "
                "from the card database instead.",
                file=sys.stderr,
            )
            return 1
        info = gather_info(driver)
        dev = info["devices"][args.device]
        page = _html.build(
            name=dev["name"],
            derived=dev["derived"],
            version=__version__,
            nvml=dev["nvml"],
            attributes=dev["attributes"],
            subtitle=f"CUDA driver {info['driver_version']}",
        )

    try:
        with open(args.out, "w") as fh:
            fh.write(page)
    except OSError as exc:
        print(f"error: can't write {args.out}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# ceiling
# ---------------------------------------------------------------------------

def cmd_ceiling(args: argparse.Namespace) -> int:
    from . import ceiling as _ceiling

    results = _ceiling.measure(mb=args.mb, matmul_n=args.matmul_n, device_index=args.device)
    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        for line in _ceiling.format_table(results):
            print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelmeter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="dump all CUDA device attributes and derived peaks")
    p_info.add_argument("--json", action="store_true", help="machine-readable output")
    p_info.set_defaults(func=cmd_info)

    p_bench = sub.add_parser("bench", help="run @kernelmeter.benchmark specs from a file")
    p_bench.add_argument("file", help="python file that registers benchmarks")
    p_bench.add_argument("--json", action="store_true", help="machine-readable output")
    p_bench.add_argument("--save", metavar="FILE", help="write results to a json file")
    p_bench.add_argument(
        "--compare", metavar="FILE", help="compare against results saved with --save"
    )
    p_bench.add_argument("--device", type=int, default=0, help="cuda device index")
    p_bench.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="skip the L2 flush between iterations (lets small workloads run hot in cache)",
    )
    p_bench.set_defaults(func=cmd_bench)

    p_roof = sub.add_parser("roofline", help="draw the device roofline")
    p_roof.add_argument("--ai", type=float, help="mark a kernel at this arithmetic intensity")
    p_roof.add_argument("--device", type=int, default=0)
    p_roof.add_argument("--gpu", help="a card from the database instead of the local device")
    p_roof.add_argument("--tensor", action="store_true", help="use the fp16 tensor-core roof")
    p_roof.add_argument("--peak-bw", type=float, help="override bandwidth in GB/s")
    p_roof.add_argument("--peak-tflops", type=float, help="override compute in TFLOP/s")
    p_roof.set_defaults(func=cmd_roofline)

    p_gpus = sub.add_parser("gpus", help="list the built-in card database")
    p_gpus.add_argument("--json", action="store_true", help="machine-readable output")
    p_gpus.set_defaults(func=cmd_gpus)

    p_cmp = sub.add_parser("compare", help="compare cards, optionally at a given intensity")
    p_cmp.add_argument("gpu", nargs="+", help="cards to compare, e.g. 4090 h100-sxm l40s")
    p_cmp.add_argument("--ai", type=float, help="your kernel's arithmetic intensity in flop/byte")
    p_cmp.add_argument("--tensor", action="store_true", help="use fp16 tensor-core roofs")
    p_cmp.add_argument(
        "--cost", help="rental prices for a TF-per-dollar column, e.g. 4090=0.44,h100-sxm=2.99"
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_llm = sub.add_parser("llm", help="token/s ceilings for llm inference on given cards")
    p_llm.add_argument("params", help="model size, e.g. 70b, 8b, 900m")
    p_llm.add_argument("--quant", default="q4", help="f16, bf16, q8, q6, q5, q4, q3 (default q4)")
    p_llm.add_argument(
        "--bytes-per-param", type=float,
        help="exact bytes per parameter, overrides --quant",
    )
    p_llm.add_argument(
        "--gpus", nargs="+",
        help="cards from the database (default: the local device)",
    )
    p_llm.add_argument("--cost", help="rental prices per card, e.g. 4090=0.35,h100-sxm=2.69")
    p_llm.add_argument("--overhead-gb", type=float, default=2.0)
    p_llm.add_argument(
        "--num-gpus", type=int, default=1,
        help="tensor-parallel split over this many identical cards",
    )
    p_llm.add_argument(
        "--active-params", help="active parameters for MoE models, e.g. 37b"
    )
    p_llm.add_argument(
        "--batch", type=int, default=1,
        help="concurrent decode streams; weights amortize until the compute roof",
    )
    p_llm.add_argument("--per-watt", action="store_true", help="add a tokens/s per watt column")
    p_llm.add_argument("--device", type=int, default=0)
    p_llm.set_defaults(func=cmd_llm)

    p_rep = sub.add_parser("report", help="write a shareable single-file html report")
    p_rep.add_argument("--out", default="kernelmeter-report.html", help="output path")
    p_rep.add_argument("--device", type=int, default=0)
    p_rep.add_argument("--gpu", help="build the report from the card database instead")
    p_rep.set_defaults(func=cmd_report)

    p_occ = sub.add_parser("occupancy", help="theoretical occupancy calculator")
    p_occ.add_argument("--block", type=int, required=True, help="threads per block")
    p_occ.add_argument("--regs", type=int, default=0, help="registers per thread (from ptxas/ncu)")
    p_occ.add_argument("--smem", type=int, default=0, help="shared memory per block in bytes")
    p_occ.add_argument("--cc", help="compute capability, e.g. 8.6 (default: ask the device)")
    p_occ.add_argument("--device", type=int, default=0)
    p_occ.set_defaults(func=cmd_occupancy)

    p_ceil = sub.add_parser("ceiling", help="measure real achievable bandwidth and FP32")
    p_ceil.add_argument("--mb", type=int, default=256, help="working-set size per array")
    p_ceil.add_argument("--matmul-n", type=int, default=4096, help="matmul size for the FP32 test")
    p_ceil.add_argument("--device", type=int, default=0, help="cuda device index")
    p_ceil.add_argument("--json", action="store_true", help="machine-readable output")
    p_ceil.set_defaults(func=cmd_ceiling)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
