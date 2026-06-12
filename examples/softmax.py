"""Row softmax: the canonical "fuse it yourself" exercise.

Run:  kernelmeter bench examples/softmax.py

A fused softmax reads each row once and writes it once. The naive
composition (exp, sum, div as separate ops) moves the data several times,
so %peak bw directly shows the value of fusion.
"""

import torch
import triton
import triton.language as tl

import kernelmeter as km

ROWS, COLS = 8192, 4096


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride + offs, out, mask=mask)


def _make_args():
    return (torch.randn(ROWS, COLS, device="cuda"),)


def _bytes(x):
    return 2 * x.numel() * x.element_size()  # one read + one write per element


@km.benchmark(
    "triton_softmax",
    args=_make_args,
    ref=lambda x: torch.softmax(x, dim=-1),
    bytes_per_call=_bytes,
)
def triton_softmax(x):
    out = torch.empty_like(x)
    block = triton.next_power_of_2(COLS)
    _softmax_kernel[(ROWS,)](x, out, x.stride(0), COLS, BLOCK=block)
    return out


@km.benchmark("torch_softmax", args=_make_args, bytes_per_call=_bytes)
def torch_softmax(x):
    return torch.softmax(x, dim=-1)
