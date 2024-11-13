import torch
from torch import Tensor
from typing import Tuple

from fused_gemm import fused_gemm_func as cutlass_fused_gemm


def run_unittest(a: Tensor,
                 b: Tensor,
                 c: Tensor,
                 d: Tensor,
                 M: int,
                 N: int,
                 K: int,
                 P: int,
                 kTM: int,
                 kTN: int,
                 kTK: int,
                 kTP: int,
                 warp_layout: Tuple,
                 debug_print=False,
                 epsilon: float = 5e-2):
    cutlass_fused_gemm(a, b, c, d, M, N, K, P, kTM, kTN, kTK, kTP, *warp_layout)
    
    ref_acc = a @ b.t()
    ref_d = ref_acc @ c.t()

    if debug_print:
        print("Result:")
        print(d)

        print("\nReference:")
        print(ref_d)

    avg_diff = (torch.sum(torch.abs(ref_d - d) / (M * N))).item()

    if avg_diff > epsilon:
        return False
    else:
        return True


def run_test(
    M: int,
    N: int,
    K: int,
    P: int, 
    kTM: int,
    kTN: int,
    kTK: int,
    kTP: int,
    warp_layout: Tuple,
):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.randn(P, N, device=device, dtype=dtype)
    d = torch.zeros(M, P, device=device, dtype=dtype)

    if run_unittest(a, b, c, d, M, N, K, P, kTM, kTN, kTK, kTP, warp_layout, debug_print=True):
        print("Unittest passed")
    else:
        raise ValueError("Unittest failed")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        cutlass_fused_gemm(a, b, c, d, M, N, K, P, kTM, kTN, kTK, kTP, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time


if __name__ == "__main__":
    kM = 1024
    kN = 1024
    kK = 1024
    kP = 1024

    kTM = 64
    kTN = 64
    kTK = 64
    kTP = 64

    time = run_test(kM, kN, kK, kP, kTM, kTN, kTK, kTP, (1, 1))

    print("Elapsed time: {:.4f} ms".format(time))
