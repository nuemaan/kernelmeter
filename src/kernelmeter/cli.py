"""kernelmeter command line interface.

  kernelmeter info               dump every device attribute + derived peaks
  kernelmeter info --json        same, machine readable
  kernelmeter bench file.py      run all @kernelmeter.benchmark specs in file
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from . import attrs as _attrs
from . import peaks as _peaks
from .cudadrv import CudaNotAvailableError, Driver


def _fmt(value: float | None, suffix: str = "", nd: int = 1) -> str:
    return f"{value:.{nd}f}{suffix}" if value is not None else "-"


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

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
        return 0

    header = (
        f"{'kernel':<24} {'median ms':>10} {'GB/s':>9} {'%peak bw':>9} "
        f"{'TFLOP/s':>9} {'%fp32':>7} {'vs ref':>8} {'correct':>8}"
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
            f"{_fmt(r.pct_peak_bw, '%'):>9} {_fmt(r.tflops, nd=2):>9} "
            f"{_fmt(r.pct_peak_fp32, '%'):>7} {speedup:>8} {correct:>8}"
        )
    return 0 if all(r.error is None and r.correct in (None, True) for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelmeter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="dump all CUDA device attributes and derived peaks")
    p_info.add_argument("--json", action="store_true", help="machine-readable output")
    p_info.set_defaults(func=cmd_info)

    p_bench = sub.add_parser("bench", help="run @kernelmeter.benchmark specs from a file")
    p_bench.add_argument("file", help="python file that registers benchmarks")
    p_bench.add_argument("--json", action="store_true", help="machine-readable output")
    p_bench.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="skip the L2 flush between iterations (lets small workloads run hot in cache)",
    )
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
