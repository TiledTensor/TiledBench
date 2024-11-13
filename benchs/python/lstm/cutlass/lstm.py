import torch
from torch import Tensor

from compile import Compile

__all__ = [
    "lstm_func",
]


class LstmFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        W: Tensor,
        X: Tensor,
        U: Tensor,
        C: Tensor,
        H: Tensor,
        CO: Tensor,
        HO: Tensor,
        M: int,
        N: int,
        K: int,
        kTM: int,
        kTN: int,
        kTK: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> Tensor:
        builder = Compile(file_prefix="lstm", tmp_dir="tmp")
        lib_name = builder.compile(M, N, K, kTM, kTN, kTK, warp_per_row,
                                   warp_per_col)

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")

        builder.apply(lib_name, [W, X, U, C, H, CO, HO], device=0)
        return C


lstm_func = LstmFunc.apply
