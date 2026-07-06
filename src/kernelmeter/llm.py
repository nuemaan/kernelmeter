"""Token-throughput ceilings for LLM inference, from roofline math.

Decode is memory-bound: generating one token reads every weight once, so
the ceiling is bandwidth / weight bytes. Prefill is compute-bound: about
2 FLOPs per parameter per token, so the ceiling is tensor throughput /
(2 x params). These are ceilings, not predictions. A well-tuned stack
(llama.cpp, vLLM, TensorRT-LLM) typically lands at 50-85% of them, and
nothing lands above.

Weight sizes use effective bytes per parameter for common quant formats
(gguf k-quants carry scales, so q4 is ~0.58, not 0.5). Pass an exact
value with --bytes-per-param when you know your file size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# effective bytes per parameter, including quantization overhead
QUANTS = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8": 1.07,
    "q6": 0.82,
    "q5": 0.70,
    "q4": 0.58,
    "q3": 0.44,
}


def parse_params(text: str) -> float:
    """'70b' -> 70e9, '1.5b' -> 1.5e9, '900m' -> 9e8, bare numbers pass through."""
    m = re.fullmatch(r"([\d.]+)\s*([bm]?)", text.strip().lower())
    if not m:
        raise ValueError(f"can't parse parameter count {text!r} (try 70b, 8b, 900m)")
    value = float(m.group(1))
    scale = {"b": 1e9, "m": 1e6, "": 1.0}[m.group(2)]
    return value * scale


def weight_bytes(params: float, bytes_per_param: float) -> float:
    return params * bytes_per_param


@dataclass
class LlmEstimate:
    name: str
    vram_gb: float | None
    fits: bool | None
    decode_tps: float | None    # ceiling, tokens/second
    prefill_tps: float | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def estimate(
    name: str,
    mem_bandwidth_gbs: float,
    compute_tflops: float | None,
    vram_bytes: float | None,
    params: float,
    bytes_per_param: float,
    overhead_gb: float = 2.0,
) -> LlmEstimate:
    """One card's ceilings for one model. compute_tflops should be the
    fp16 tensor peak when known, else fp32. vram_bytes None means the
    capacity is unknown and the fit check is skipped."""
    weights = weight_bytes(params, bytes_per_param)

    fits = None
    if vram_bytes is not None:
        fits = weights + overhead_gb * 1e9 <= vram_bytes

    decode = prefill = None
    if fits is not False:
        decode = mem_bandwidth_gbs * 1e9 / weights
        if compute_tflops:
            prefill = compute_tflops * 1e12 / (2.0 * params)

    return LlmEstimate(
        name=name,
        vram_gb=vram_bytes / 1e9 if vram_bytes is not None else None,
        fits=fits,
        decode_tps=decode,
        prefill_tps=prefill,
    )
