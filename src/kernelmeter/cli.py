"""kernelmeter command line interface.

  kernelmeter info                  dump every device attribute + derived peaks
  kernelmeter bench file.py         run all @kernelmeter.benchmark specs in file
  kernelmeter roofline              draw the device roofline, place your kernel on it
  kernelmeter occupancy             theoretical occupancy from block/regs/smem
  kernelmeter ceiling               measure real achievable bandwidth and FP32
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from . import attrs as _attrs
from . import occupancy as _occupancy
from . import peaks as _peaks
from . import roofline as _roofline
from .cudadrv import CudaNotAvailableError, Driver


def _fmt(value: float | None, suffix: str = "", nd: int = 1) -> str:
    return f"{value:.{nd}f}{suffix}" if value is not None else "-"


def _device_attrs(ordinal: int = 0) -> dict[str, int]:
    driver = Driver()
    return _attrs.query_all(driver, driver.device(ordinal))


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def gather_info(driver: Driver) -> dict:
    major, minor = driver.driver_version()
    devices = []
    for ordinal in range(driver.device_count()):
        dev = driver.device(ordinal)
        attributes = _attrs.query_all(driver, dev)
        peaks = _peaks.derive(attributes)
        devices.append(
            {
                "ordinal": ordinal,
                "name": dev.name,
                "total_memory_bytes": dev.total_mem_bytes,
                "derived": peaks.as_dict(),
                "attributes": attributes,
            }
        )
    return {"driver_version": f"{major}.{minor}", "devices": devices}


def cmd_info(args: argparse.Namespace) -> int:
    try:
        driver = Driver()
    except CudaNotAvailableError as exc:
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

    results = _bench.run_registry(flush_l2=not args.no_flush_l2)

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
    if peak_bw is None or peak_tf is None:
        try:
            driver = Driver()
            dev = driver.device(args.device)
            name = dev.name
            peaks = _peaks.derive(_attrs.query_all(driver, dev))
            peak_bw = peak_bw or peaks.mem_bandwidth_gbs
            peak_tf = peak_tf or peaks.fp32_tflops
        except CudaNotAvailableError:
            pass
    if not peak_bw or not peak_tf:
        print(
            "error: no CUDA device found. Pass --peak-bw GB/s and "
            "--peak-tflops to draw a roofline for any card.",
            file=sys.stderr,
        )
        return 1

    ridge = _roofline.ridge_point(peak_tf, peak_bw)
    if name:
        print(f"Device {args.device}: {name}")
    print(f"  peak bandwidth : {peak_bw:.1f} GB/s")
    print(f"  peak compute   : {peak_tf:.2f} TFLOP/s")
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

    occ = _occupancy.compute(args.block, args.regs, args.smem, limits)
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
# ceiling
# ---------------------------------------------------------------------------

def cmd_ceiling(args: argparse.Namespace) -> int:
    from . import ceiling as _ceiling

    results = _ceiling.measure(mb=args.mb, matmul_n=args.matmul_n)
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
    p_bench.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="skip the L2 flush between iterations (lets small workloads run hot in cache)",
    )
    p_bench.set_defaults(func=cmd_bench)

    p_roof = sub.add_parser("roofline", help="draw the device roofline")
    p_roof.add_argument("--ai", type=float, help="mark a kernel at this arithmetic intensity")
    p_roof.add_argument("--device", type=int, default=0)
    p_roof.add_argument("--peak-bw", type=float, help="override bandwidth in GB/s")
    p_roof.add_argument("--peak-tflops", type=float, help="override compute in TFLOP/s")
    p_roof.set_defaults(func=cmd_roofline)

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
    p_ceil.add_argument("--json", action="store_true", help="machine-readable output")
    p_ceil.set_defaults(func=cmd_ceiling)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
