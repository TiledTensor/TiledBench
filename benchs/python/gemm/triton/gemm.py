import torch 
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TRITON_INTERPRET"] = '1'

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8)
    ],
    key = ['M', 'N', 'K']
)

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offset_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offset_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask = offset_k[None, :] < K - k * BLOCK_K)
        b = tl.load(b_ptrs, mask = offset_k[:, None] < K - k * BLOCK_K)
        
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
        
    offset_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn)
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    
    tl.store(c_ptrs, acc, mask = c_mask)
    
def gemm(a, b):
    assert a.shape[1] == b.shape[0], "shape mismatch"
    assert a.is_contiguous() and b.is_contiguous(), "input must be contiguous"
    
    M, K = a.shape 
    K, N = b.shape
    
    c = torch.empty((M, N), device = a.device, dtype = torch.float16)
    
    def grid(META):
        return (tl.cdiv(M, META['BLOCK_M']), tl.cdiv(N, META['BLOCK_N']), 1)
    
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),  
        c.stride(0), c.stride(1)
    )

    return c
