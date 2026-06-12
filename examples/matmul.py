"""Matmul: a compute-bound kernel, scored against the roofline.

Run:  kernelmeter bench examples/matmul.py

At n=4096 a matmul does roughly 680 flops per byte moved, far past the
ridge point of any card, so the %roof column scores it against the FP32
compute peak instead of bandwidth. TF32 is disabled so cuBLAS stays on
the plain FP32 units and the comparison is fair.
"""

import torch

import kernelmeter as km

N = 4096
torch.backends.cuda.matmul.allow_tf32 = False


def _make_args():
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")
    return (a, b)


@km.benchmark(
    "fp32_matmul",
    args=_make_args,
    flops_per_call=2 * N**3,
    bytes_per_call=3 * N * N * 4,
)
def fp32_matmul(a, b):
    return a @ b
