"""PMPP chapter 2 classic: vector addition, benchmarked honestly.

Run:  kernelmeter bench examples/vector_add.py

Vector add is memory-bound: it reads 2 floats and writes 1 per element, so
the interesting number is %peak bw, not TFLOP/s. A good implementation
should land above ~80% of theoretical bandwidth on most cards.
"""

import torch
import triton
import triton.language as tl

import kernelmeter as km

N = 1 << 26  # 64M elements -> ~768 MB moved per call, far bigger than L2


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _make_args():
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    return (x, y)


@km.benchmark(
    "triton_vector_add",
    args=_make_args,
    ref=torch.add,
    bytes_per_call=lambda x, y: 3 * x.numel() * x.element_size(),
)
def triton_vector_add(x, y):
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, 1024),)
    _add_kernel[grid](x, y, out, N, BLOCK=1024)
    return out


@km.benchmark(
    "torch_vector_add",
    args=_make_args,
    bytes_per_call=lambda x, y: 3 * x.numel() * x.element_size(),
)
def torch_vector_add(x, y):
    return x + y
