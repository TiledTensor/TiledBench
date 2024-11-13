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

def run_cublas_unittest(
        a: Tensor,
        b: Tensor,
        c: Tensor,
        M: int,
        N: int,
        K: int,
        debug_print=False,
        epsilon: float = 5e-2
):
    time = torch.zeros(1, device=torch.device("cpu"), dtype=torch.float32)
    cublas_gemm(M, N, K, a, b, c, time)
    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        print("\nReference:")
        print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()

    if avg_diff > epsilon:
        return False
    else:
        return True

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

def run_cutlass_unittest(
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
        debug_print=False,
        epsilon: float = 5e-2
):
    cutlass_gemm(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)
    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        print("\nReference:")
        print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c) / (M * N))).item()

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
    if run_cublas_unittest(a, b, c, M, N, K):
        pass
    else:
        print("Run cuBLAS unittest failed")
        return float("NaN")

    iters = 50
    warmup = 10

    cublas_gemm(M, N, K, a, b, c, time, iters, warmup)

    return time.item()

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
    if run_cutlass_unittest(a, b, c, M, N, K, kTM, kTN, kTK, warp_layout):
        pass
    else:
        print("Run Cutlass unittest failed")
        return float("NaN")

    warmup = 10
    iters = 50

    for _ in range(warmup):
        cutlass_gemm(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)
    

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

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
        pass
    else:
        print("Run TiledCUDA unittest failed")
        return float('NaN')

    warmup = 10
    iters = 50

    for _ in range(warmup):
        tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        tiledcuda_gemm(a, b, c, M, N, K, kTM, kTN, kTK, kRK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time

def run_bench(
    M: int,
    N: int,
    K: int,
    kTM: int,
    kTN: int,
    kTK: int,
    kRK: int,
    warp_layout: Tuple,
    record_csv = None
):  
    torch.manual_seed(1234)

    a = torch.randn(M, K, device = 'cuda', dtype = torch.float16)
    b = torch.randn(N, K, device = 'cuda', dtype = torch.float16)
    c = torch.zeros(M, N, device = 'cuda', dtype = torch.float32)
    half_c = torch.zeros(M, N, device = 'cuda', dtype = torch.float16)
    
    cublas_time_tensor = torch.zeros(1, device=torch.device("cpu"), dtype=torch.float32)

    cublas_time = run_cublas_bench(a, b, half_c, M, N, K, cublas_time_tensor)
    cutlass_time = run_cutlass_bench(a, b, half_c, M, N, K, kTM, kTN, kTK, warp_layout)
    tiledcuda_time = run_tiledcuda_bench(a, b, c, M, N, K, kTM, kTN, kTK, kRK, warp_layout)

    print("(M, N, K) (kTM, kTN, kTK)")
    print("({}, {}, {}) ({}, {}, {})".format(M, N, K, kTM, kTN, kTK))
    print("cublas_time: {:.4f} ms, cutlass_time: {:.4f} ms, tiledcuda_time: {:.4f} ms".format(cublas_time, cutlass_time, tiledcuda_time))

    csv.writer(record_csv).writerow([M, N, K, kTM, kTN, kTK, "{:.4f}".format(cublas_time), "{:.4f}".format(cutlass_time), "{:.4f}".format(tiledcuda_time)])


if __name__ == "__main__":
    kRK = 32

    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id).replace(" ", "_")

    record = 'gemm_bench_{}.csv'.format(device_name)
    record_csv = open(record, 'w', newline='')  

    csv.writer(record_csv).writerow(["M", "N", "K", "kTM", "kTN", "kTK", "cuBLAS(ms)", "Cutlass(ms)", "TiledCUDA(ms)"])

    run_bench(4096, 4096, 2048, 128, 256, 64, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 64, 256, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 128, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 64, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 64, 128, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 32, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 32, 64, 32, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 256, 128, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 256, 128, 128, kRK, (2, 2), record_csv)
    # Cutlass RuntimeError: CUDA error: an illegal memory access was encountered.
    # run_bench(4096, 4096, 2048, 256, 64, 128, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 64, 256, 128, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 128, 128, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 128, 64, 64, kRK, (2, 2), record_csv)
    run_bench(4096, 4096, 2048, 64, 128, 64, kRK, (2, 2), record_csv)
    # Cutlass RuntimeError: CUDA error: an illegal memory access was encountered.
    # run_bench(4096, 4096, 2048, 128, 32, 64, kRK, (2, 2), record_csv)

    record_csv.close()
