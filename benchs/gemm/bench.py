import torch
from torch import Tensor
from typing import Tuple
import sys
import os

cutlass_dir = os.path.join(os.path.dirname(__file__), 'cutlass')
sys.path.insert(0, cutlass_dir)

tiledcuda_dir = os.path.join(os.path.dirname(__file__), 'tiledcuda')
sys.path.insert(0, tiledcuda_dir)

from cutlass.gemm import gemm_func as cutlass_gemm
from tiledcuda.gemm import gemm_func as tiledcuda_gemm
from cuBLAS import cublas_gemm



def run_tiledcuda_unittest(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        M: int,
        N: int,
        K: int,
        kTM: int,
        kTN: int,
        kTK: int,
        kRK: int,
        warp_layout: Tuple,
        debug_print=False,
        epsilon: float = 5e-2
):
    tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        print("\nReference:")
        print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c.half()) / (M * N))).item()

    if avg_diff > epsilon:
        return False
    else:
        return True

def run_cublas_bench(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    M: int,
    N: int,
    K: int,
    time: Tensor
):
    warmup = 10
    iters = 50
    cublas_gemm(M, N, K, a, b, c, time, iters, warmup)

    return time

def run_cutlass_bench(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    M: int,
    N: int,
    K: int,
    kTM: int,
    kTN: int,
    kTK: int,
    warp_layout: Tuple,
):

    # if run_unittest(a, b, c, M, N, K, kTM, kTN, kTK, warp_layout):
    #     print("Unittest passed")
    # else:
    #     raise ValueError("Unittest failed")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        cutlass_gemm(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time


def run_tiledcuda_bench(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    M: int,
    N: int,
    K: int,
    kTM: int,
    kTN: int,
    kTK: int,
    kRK: int,
    warp_layout: Tuple,
):

    if run_tiledcuda_unittest(a, b, c, M, N, K, kTM, kTN, kTK, kRK, warp_layout):
        print("Unittest passed")
    else:
        raise ValueError("Unittest failed")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time


if __name__ == "__main__":
    kM = 4096
    kN = 4096
    kK = 2048

    torch.manual_seed(1234)

    a = torch.randn(kM, kK, device=torch.device("cuda"), dtype=torch.float16)
    b = torch.randn(kN, kK, device=torch.device("cuda"), dtype=torch.float16)
    c = torch.zeros(kM, kN, device=torch.device("cuda"), dtype=torch.float32)

    kTM = 64
    kTN = 256
    kTK = 32

    kRK = 32

    cublas_time = torch.zeros(1, device=torch.device("cpu"), dtype=torch.float32)
    
    cublas_gemm = run_cublas_bench(a, b, c, kM, kN, kK, cublas_time)
    cutlass_time = run_cutlass_bench(a, b, c, kM, kN, kK, kTM, kTN, kTK, (2, 2))
    tiledcuda_time = run_tiledcuda_bench(a, b, c, kM, kN, kK, kTM, kTN, kTK, kRK, (2, 2))

    # print("Elapsed time: {:.4f} ms".format(time))

    # print(f"cutlass_time: {cutlass_time}, tiledcuda_time: {tiledcuda_time}")
    print("(kM, kN, kK) (kTM, kTN, kTK)")
    print("({}, {}, {}) ({}, {}, {})".format(kM, kN, kK, kTM, kTN, kTK))
    print("cublas_time: {:.4f} ms, cutlass_time: {:.4f} ms, tiledcuda_time: {:.4f} ms".format(cublas_time.item(), cutlass_time, tiledcuda_time))