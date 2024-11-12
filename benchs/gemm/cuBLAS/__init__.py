import torch
from torch import Tensor
import os

dyn_lib_path = os.path.join(os.path.dirname(__file__), "src/build/libcublas_gemm.so")

torch.ops.load_library(dyn_lib_path)

def cublas_gemm(
        m: int,
        n: int,
        k: int,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        elapsed_time: Tensor,
        iters: int = 0,
        warmup: int = 0
):
    torch.ops.cublas_gemm.gemm(m, n, k, a, b, c, elapsed_time, iters, warmup)
