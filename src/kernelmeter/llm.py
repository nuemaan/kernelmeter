"""Token-throughput ceilings for LLM inference, from roofline math.

Decode is memory-bound at batch 1: generating one token reads every
active weight once, so the ceiling is bandwidth / read bytes. Batching
amortizes the weight reads across streams until the compute roof takes
over (about 2 FLOPs per active parameter per token), which is the same
ridge-point logic as any other roofline. Prefill is the compute-bound
case outright.

These are ceilings, not predictions. A well-tuned stack (llama.cpp,
vLLM, TensorRT-LLM) typically lands at 50-85% of them, and nothing lands
above. Multi-GPU splits assume an ideal tensor-parallel split; real
interconnects cost a further slice.

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
    decode_tps: float | None       # total tokens/second ceiling across the batch
    per_stream_tps: float | None   # per request
    prefill_tps: float | None
    bound: str | None              # what limits the decode step: "memory" or "compute"

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
    active_params: float | None = None,
    num_gpus: int = 1,
    batch: int = 1,
) -> LlmEstimate:
    """One setup's ceilings for one model.

    compute_tflops is the tensor peak at the accumulate width the stack
    really uses (halve GeForce/Titan fp16-acc numbers before passing).
    active_params covers MoE: fit uses the full parameter count, per-token
    reads and FLOPs use the active count. num_gpus splits the weights and
    aggregates bandwidth/compute, assuming an ideal tensor-parallel split.
    """
    weights_total = weight_bytes(params, bytes_per_param)
    active = active_params if active_params is not None else params
    read_bytes = weight_bytes(active, bytes_per_param)

    fits = None
    if vram_bytes is not None:
        fits = weights_total / num_gpus + overhead_gb * 1e9 <= vram_bytes

    decode = per_stream = prefill = None
    bound = None
    if fits is not False:
        bw = mem_bandwidth_gbs * 1e9 * num_gpus
        step_mem = read_bytes / bw
        step_comp = 0.0
        if compute_tflops:
            comp = compute_tflops * 1e12 * num_gpus
            step_comp = 2.0 * active * batch / comp
            prefill = comp / (2.0 * active)
        step = max(step_mem, step_comp)
        decode = batch / step
        per_stream = 1.0 / step
        bound = "compute" if step_comp > step_mem else "memory"

    return LlmEstimate(
        name=name,
        vram_gb=vram_bytes / 1e9 if vram_bytes is not None else None,
        fits=fits,
        decode_tps=decode,
        per_stream_tps=per_stream,
        prefill_tps=prefill,
        bound=bound,
    )
