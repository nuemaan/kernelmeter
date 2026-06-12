"""Roofline model: how fast a kernel *can* go at a given arithmetic intensity.

A kernel that does f FLOPs while moving b bytes has intensity f/b. Below
the ridge point the memory system is the limit, above it the ALUs are.
The attainable ceiling at intensity I is min(peak_flops, I * peak_bw),
and a kernel's quality is best judged against that, not against whichever
single peak happens to look more flattering.
"""

from __future__ import annotations

import math


def intensity(flops: int, nbytes: int) -> float:
    return flops / nbytes


def ridge_point(peak_tflops: float, peak_bw_gbs: float) -> float:
    """Intensity (flop/byte) where the memory roof meets the compute roof."""
    return 1000.0 * peak_tflops / peak_bw_gbs


def attainable_tflops(ai: float, peak_tflops: float, peak_bw_gbs: float) -> float:
    return min(peak_tflops, ai * peak_bw_gbs / 1000.0)


def bound(ai: float, peak_tflops: float, peak_bw_gbs: float) -> str:
    return "mem" if ai < ridge_point(peak_tflops, peak_bw_gbs) else "comp"


def render(
    peak_tflops: float,
    peak_bw_gbs: float,
    ai: float | None = None,
    width: int = 58,
    height: int = 12,
) -> list[str]:
    """Draw the roofline as text. Log-log, x from 1/8 to 256 flop/byte.

    The roof is drawn with '*', the ridge column is marked 'x', and the
    optional ai argument puts an 'o' where the caller's kernel sits.
    """
    lo, hi = -3.0, 8.0  # log2 of the intensity axis
    ridge = ridge_point(peak_tflops, peak_bw_gbs)

    def col_to_ai(c: int) -> float:
        return 2.0 ** (lo + (hi - lo) * c / (width - 1))

    def ai_to_col(a: float) -> int:
        c = round((math.log2(a) - lo) / (hi - lo) * (width - 1))
        return min(max(c, 0), width - 1)

    ys = [attainable_tflops(col_to_ai(c), peak_tflops, peak_bw_gbs) for c in range(width)]
    ymin, ymax = math.log10(ys[0]), math.log10(peak_tflops)

    def y_to_row(y: float) -> int:
        if ymax == ymin:
            return height - 1
        frac = (math.log10(y) - ymin) / (ymax - ymin)
        return min(max(round(frac * (height - 1)), 0), height - 1)

    grid = [[" "] * width for _ in range(height)]
    for c, y in enumerate(ys):
        grid[height - 1 - y_to_row(y)][c] = "*"
    grid[height - 1 - y_to_row(attainable_tflops(ridge, peak_tflops, peak_bw_gbs))][
        ai_to_col(ridge)
    ] = "x"
    if ai is not None:
        a = min(max(ai, 2.0**lo), 2.0**hi)
        grid[height - 1 - y_to_row(attainable_tflops(a, peak_tflops, peak_bw_gbs))][
            ai_to_col(a)
        ] = "o"

    top_label = f"{peak_tflops:.2f} TF/s "
    pad = len(top_label)
    lines = []
    for r, row in enumerate(grid):
        label = top_label if r == 0 else " " * pad
        lines.append(label + "|" + "".join(row))
    lines.append(" " * pad + "+" + "-" * width)
    ticks = {ai_to_col(2.0**e): f"2^{e}" for e in (-3, 0, 3, 6)}
    axis = [" "] * (width + 1)
    for col, text in ticks.items():
        for i, ch in enumerate(text):
            if col + i < len(axis):
                axis[col + i] = ch
    lines.append(" " * pad + " " + "".join(axis) + " flop/byte")
    return lines
