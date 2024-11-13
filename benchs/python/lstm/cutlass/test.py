import torch
from torch import Tensor
from typing import Tuple

from lstm import lstm_func as cutlass_lstm


def run_unittest(w: Tensor,
                 x: Tensor,
                 u: Tensor,
                 c0: Tensor,
                 h0: Tensor,
                 c1: Tensor,
                 h1: Tensor,
                 M: int,
                 N: int,
                 K: int,
                 kTM: int,
                 kTN: int,
                 kTK: int,
                 warp_layout: Tuple,
                 debug_print=False,
                 epsilon: float = 5e-2):
    wdata = w.flatten()
    xdata = x.t().flatten()
    udata = u.flatten()
    c0data = c0.flatten()
    h0data = h0.t().flatten()
    c1data = c1.flatten()
    h1data = h1.flatten()
    
    cutlass_lstm(wdata, xdata, udata, c0data, h0data, c1data, h1data, M, N, K, kTM, kTN, kTK, *warp_layout)
    
    # Input Gate
    i = torch.sigmoid(
        w[0] @ x + u[0] @ h0
    )
    # Forget Gate
    f = torch.sigmoid(
        w[1] @ x + u[1] @ h0
    )
    # Output Gate
    o = torch.sigmoid(
        w[2] @ x + u[2] @ h0
    )
    # Cell Gate
    c = torch.tanh(
        w[3] @ x + u[3] @ h0
    )
    
    ref_c1 = f * c0 + i * c
    ref_h1 = o * torch.tanh(c1)
    c1 = c1.view(hidden_size, batch_size)
    h1 = h1.view(hidden_size, batch_size)

    if debug_print:
        print("Result:")
        print("c: ", c1)
        print("h: ", h1)

        print("\nReference:")
        print("c: ", ref_c1)
        print("h: ", ref_h1)

    avg_diff = (torch.sum(torch.abs(ref_c1 - c1) / (M * N))).item()

    if avg_diff > epsilon:
        print(f"Average difference: {avg_diff}")
        return False
    else:
        return True


def run_test(
    hidden_size: int,
    batch_size: int,
    kTM: int,
    kTN: int,
    kTK: int,
    warp_layout: Tuple,
):
    
    
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)
    
    M = 4 * hidden_size
    N = batch_size
    K = hidden_size

    w = torch.randn(4, hidden_size, hidden_size, device=device, dtype=dtype)
    x = torch.randn(hidden_size, batch_size, device=device, dtype=dtype)
    u = torch.randn(4, hidden_size, hidden_size, device=device, dtype=dtype)
    c0 = torch.randn(hidden_size, batch_size, device=device, dtype=dtype)
    h0 = torch.randn(hidden_size, batch_size, device=device, dtype=dtype)
    c1 = torch.empty(hidden_size, batch_size, device=device, dtype=dtype)
    h1 = torch.empty(hidden_size, batch_size, device=device, dtype=dtype)

    if run_unittest(w, x, u, c0, h0, c1, h1, M, N, K, kTM, kTN, kTK, warp_layout, debug_print=True):
        print("Unittest passed")
    else:
        raise ValueError("Unittest failed")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        cutlass_lstm(w, x, u, c0, h0, c1, h1, M, N, K, kTM, kTN, kTK, *warp_layout)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time


if __name__ == "__main__":
    hidden_size = 1024
    batch_size = 256

    kTM = 32
    kTN = 32
    kTK = 32

    time = run_test(hidden_size, batch_size, kTM, kTN, kTK, (2, 2))

    print("Elapsed time: {:.4f} ms".format(time))
