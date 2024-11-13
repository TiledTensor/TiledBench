import torch
from torch import Tensor
from typing import Tuple

from batched_gemm import batched_gemm_func as cutlass_batched_gemm


def run_unittest(a: Tensor,
                 b: Tensor,
                 c: Tensor,
                 M: int,
                 N: int,
                 K: int,
                 BatchCount: int,
                 kTM: int,
                 kTN: int,
                 kTK: int,
                 warp_layout: Tuple,
                 debug_print=False,
                 epsilon: float = 5e-2):
    cutlass_batched_gemm(a, b, c, M, N, K, BatchCount, kTM, kTN, kTK, *warp_layout)
    ref_c = torch.bmm(a.view(BatchCount, M, K), b.transpose(1, 2).view(BatchCount, K, N))

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


def run_test(
    M: int,
    N: int,
    K: int,
    BatchCount: int,
    kTM: int,
    kTN: int,
    kTK: int,
    warp_layout: Tuple,
):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.randn(BatchCount, M, K, device=device, dtype=dtype)
    b = torch.randn(BatchCount, N, K, device=device, dtype=dtype)
    c = torch.zeros(BatchCount, M, N, device=device, dtype=dtype)

    if run_unittest(a, b, c, M, N, K, BatchCount, kTM, kTN, kTK, warp_layout,       debug_print=True):
        print("Unittest passed")
    else:
        raise ValueError("Unittest failed")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        cutlass_batched_gemm(a, b, c, M, N, K, BatchCount, kTM, kTN, kTK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time


if __name__ == "__main__":
    kM = 256
    kN = 256
    kK = 256
    BatchCount = 10

    kTM = 32
    kTN = 32
    kTK = 32

    time = run_test(kM, kN, kK, BatchCount, kTM, kTN, kTK, (2, 2))

    print("Elapsed time: {:.4f} ms".format(time))
