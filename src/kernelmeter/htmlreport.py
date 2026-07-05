"""Render a device (or database card) as a single self-contained HTML file.

No javascript, no external assets: one file you can attach to an issue,
drop in a wiki, or screenshot. The roofline is inline SVG built from the
same numbers the terminal chart uses.
"""

from __future__ import annotations

import html
import math

from . import roofline as _roofline


def svg_roofline(peak_tflops: float, peak_bw_gbs: float, width: int = 640, height: int = 320) -> str:
    lo, hi = -3.0, 8.0
    pad_l, pad_b, pad_t, pad_r = 64, 40, 16, 16
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    ys = [
        _roofline.attainable_tflops(2.0 ** (lo + (hi - lo) * i / 199), peak_tflops, peak_bw_gbs)
        for i in range(200)
    ]
    ymin, ymax = math.log10(ys[0]), math.log10(peak_tflops)

    def x_px(i: int) -> float:
        return pad_l + plot_w * i / 199

    def y_px(y: float) -> float:
        frac = (math.log10(y) - ymin) / (ymax - ymin) if ymax > ymin else 1.0
        return pad_t + plot_h * (1 - frac)

    points = " ".join(f"{x_px(i):.1f},{y_px(y):.1f}" for i, y in enumerate(ys))
    ridge = _roofline.ridge_point(peak_tflops, peak_bw_gbs)
    ridge_x = pad_l + plot_w * (math.log2(ridge) - lo) / (hi - lo)

    ticks = []
    for e in (-3, 0, 3, 6):
        tx = pad_l + plot_w * (e - lo) / (hi - lo)
        ticks.append(
            f'<text x="{tx:.0f}" y="{height - 14}" fill="#8b949e" '
            f'font-size="12" text-anchor="middle">2^{e}</text>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">
  <rect width="{width}" height="{height}" fill="#0d1117"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#30363d"/>
  <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#30363d"/>
  <line x1="{ridge_x:.1f}" y1="{pad_t}" x2="{ridge_x:.1f}" y2="{height - pad_b}" stroke="#30363d" stroke-dasharray="4 4"/>
  <polyline points="{points}" fill="none" stroke="#58a6ff" stroke-width="2.5"/>
  <text x="{pad_l - 8}" y="{pad_t + 12}" fill="#8b949e" font-size="12" text-anchor="end">{peak_tflops:.1f} TF/s</text>
  <text x="{ridge_x:.0f}" y="{pad_t + 12}" fill="#8b949e" font-size="12" text-anchor="middle">ridge {ridge:.1f}</text>
  <text x="{width / 2:.0f}" y="{height - 2}" fill="#8b949e" font-size="12" text-anchor="middle">flop/byte</text>
  {''.join(ticks)}
</svg>"""


def _tile(label: str, value: str) -> str:
    return (
        f'<div class="tile"><div class="v">{html.escape(value)}</div>'
        f'<div class="l">{html.escape(label)}</div></div>'
    )


def _fmt(value, suffix: str = "", nd: int = 1) -> str:
    return f"{value:.{nd}f}{suffix}" if value is not None else "-"


def build(
    name: str,
    derived: dict,
    version: str,
    nvml: dict | None = None,
    attributes: dict | None = None,
    subtitle: str = "",
) -> str:
    """Assemble the report. `derived` is a Peaks.as_dict(); nvml and
    attributes are optional and simply omitted when absent."""
    bw = derived.get("theoretical_mem_bandwidth_gb_s")
    tf = derived.get("theoretical_fp32_tflops")
    fp16 = derived.get("theoretical_fp16_tensor_tflops")

    tiles = [
        _tile("compute capability", str(derived.get("compute_capability") or "-")),
        _tile("mem bandwidth", _fmt(bw, " GB/s")),
        _tile("fp32 peak", _fmt(tf, " TFLOP/s", nd=2)),
    ]
    if fp16:
        tiles.append(_tile("fp16 tensor (dense)", _fmt(fp16, " TFLOP/s", nd=2)))
    if bw and tf:
        from . import roofline as _r

        tiles.append(_tile("ridge point", f"{_r.ridge_point(tf, bw):.1f} flop/byte"))

    facts = ""
    if nvml:
        rows = []
        if nvml.get("architecture"):
            rows.append(("architecture", nvml["architecture"]))
        if nvml.get("num_gpu_cores"):
            rows.append(("cuda cores", str(nvml["num_gpu_cores"])))
        if nvml.get("pcie_gen_max"):
            rows.append(
                ("pcie link",
                 f"gen{nvml.get('pcie_gen_current')}/{nvml['pcie_gen_max']} "
                 f"x{nvml.get('pcie_width_current')}/{nvml.get('pcie_width_max')}"),
            )
        if nvml.get("ecc_enabled") is not None:
            rows.append(("ecc", "on" if nvml["ecc_enabled"] else "off"))
        if nvml.get("vbios_version"):
            rows.append(("vbios", nvml["vbios_version"]))
        if nvml.get("driver_version"):
            rows.append(("driver", nvml["driver_version"]))
        if rows:
            body = "".join(
                f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows
            )
            facts = f"<h2>device</h2><table>{body}</table>"

    attr_rows = ""
    if attributes:
        keep = (
            "multiprocessor_count", "max_threads_per_block",
            "max_threads_per_multiprocessor", "max_blocks_per_multiprocessor",
            "max_shared_memory_per_block_optin", "max_registers_per_multiprocessor",
            "l2_cache_size", "warp_size",
        )
        body = "".join(
            f"<tr><td>{k}</td><td>{attributes[k]}</td></tr>" for k in keep if k in attributes
        )
        if body:
            attr_rows = f"<h2>limits</h2><table>{body}</table>"

    svg = svg_roofline(tf, bw) if tf and bw else ""
    sub = f'<p class="sub">{html.escape(subtitle)}</p>' if subtitle else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} - kernelmeter</title>
<style>
body{{background:#0d1117;color:#e6edf3;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.6rem;margin:0}} h2{{font-size:1rem;color:#8b949e;margin:1.6rem 0 .4rem;text-transform:lowercase}}
.sub{{color:#8b949e;margin:.2rem 0 0}}
.tiles{{display:flex;flex-wrap:wrap;gap:.6rem;margin:1.2rem 0}}
.tile{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.7rem 1rem;min-width:120px}}
.tile .v{{font-size:1.15rem;font-weight:600}} .tile .l{{font-size:.75rem;color:#8b949e}}
table{{border-collapse:collapse;width:100%}} td{{padding:.3rem .5rem;border-bottom:1px solid #21262d}}
td:first-child{{color:#8b949e}}
svg{{width:100%;height:auto;border:1px solid #30363d;border-radius:8px;margin-top:.4rem}}
footer{{color:#8b949e;font-size:.8rem;margin:2rem 0 1rem}}
a{{color:#58a6ff;text-decoration:none}}
</style></head>
<body>
<h1>{html.escape(name)}</h1>{sub}
<div class="tiles">{''.join(tiles)}</div>
<h2>roofline</h2>
{svg}
{facts}
{attr_rows}
<footer>generated by <a href="https://github.com/nuemaan/kernelmeter">kernelmeter</a> {html.escape(version)}</footer>
</body></html>"""
