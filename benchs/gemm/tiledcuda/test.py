import torch
from torch import Tensor
from typing import Tuple

from gemm import gemm_func as tiledcuda_gemm


def run_unittest(a: Tensor,
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
                 epsilon: float = 5e-2):
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


def run_test(
    M: int,
    N: int,
    K: int,
    kTM: int,
    kTN: int,
    kTK: int,
    kRK: int,
    warp_layout: Tuple,
):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    if run_unittest(a, b, c, M, N, K, kTM, kTN, kTK, kRK, warp_layout):
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

    kTM = 64
    kTN = 256
    kTK = 32

    kRK = 32

    time = run_test(kM, kN, kK, kTM, kTN, kTK, kRK, (2, 2))

    print("Elapsed time: {:.4f} ms".format(time))
