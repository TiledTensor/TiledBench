import torch
from torch import Tensor
from typing import Tuple
import sys
import os
import csv

cutlass_dir = os.path.join(os.path.dirname(__file__), 'cutlass')
sys.path.insert(0, cutlass_dir)

tiledcuda_dir = os.path.join(os.path.dirname(__file__), 'tiledcuda')
sys.path.insert(0, tiledcuda_dir)

from cutlass.gemm import gemm_func as cutlass_gemm
from tiledcuda.gemm import gemm_func as tiledcuda_gemm
from cuBLAS import cublas_gemm

def run_cutlass_unittest(
        M: int,
        N: int,
        K: int,
        kTM: int,
        kTN: int,
        kTK: int,
        warp_layout: Tuple,
        debug_print=False,
        epsilon: float = 5e-2
):
    a = torch.randn(M, K, device=torch.device("cuda"), dtype=torch.float16)
    b = torch.randn(N, K, device=torch.device("cuda"), dtype=torch.float16)
    c = torch.zeros(M, N, device=torch.device("cuda"), dtype=torch.float16)

    cutlass_gemm(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)
    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        print("\nReference:")
        print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()

    if avg_diff > epsilon:
        print("({}, {}, {}) ({}, {}, {}) run unittest failed".format(M, N, K, kTM, kTN, kTK))
    else:
        print("({}, {}, {}) ({}, {}, {}) run unittest passed".format(M, N, K, kTM, kTN, kTK))


if __name__ == "__main__":
    run_cutlass_unittest(4096, 4096, 2048, 128, 256, 64, (2, 2), True)


