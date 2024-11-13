import torch
from torch import Tensor

from compile import Compile

__all__ = [
    "batched_gemm_func",
]


class BatchedGemmFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        M: int,
        N: int,
        K: int,
        BatchCount: int,
        kTM: int,
        kTN: int,
        kTK: int,
        warp_per_row: int,
        warp_per_col: int,
    ) -> Tensor:
        builder = Compile(file_prefix="batched_gemm", tmp_dir="tmp")
        lib_name = builder.compile(M, N, K, BatchCount, kTM, kTN, kTK, warp_per_row,
                                   warp_per_col)

        if lib_name is None:
            raise RuntimeError("Failed to compile the library.")

        builder.apply(lib_name, [A, B, C], device=0)
        return C


batched_gemm_func = BatchedGemmFunc.apply
