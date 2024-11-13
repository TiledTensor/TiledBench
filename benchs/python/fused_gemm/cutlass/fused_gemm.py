import torch
from torch import Tensor

from compile import Compile

__all__ = [
    "fused_gemm_func",
]


class FusedGemmFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        M: int,
        N: int,
        K: int,
        P: int,
        kTM: int,
        kTN: int,
        kTK: int,
        kTP: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> Tensor:
        builder = Compile(file_prefix="fused_gemm", tmp_dir="tmp")
        lib_name = builder.compile(M, N, K, P, kTM, kTN, kTK, kTP, warp_per_row,
                                   warp_per_col)

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")
        

        builder.apply(lib_name, [A, B, C, D], device=0)
        return D


fused_gemm_func = FusedGemmFunc.apply
