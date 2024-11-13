entry = """#include "../cutlass_fused_gemm.cuh"

extern "C" int kernel_entry(const __half* dA, const __half* dB, const __half* dC, __half* dD) {{
    using DType = cutlass::half_t;

    auto* A = reinterpret_cast<const DType*>(dA);
    auto* B = reinterpret_cast<const DType*>(dB);
    auto* C = reinterpret_cast<const DType*>(dC);
    auto* D = reinterpret_cast<DType*>(dD);

    cute_fused_gemm<DType, {WarpPerRow}, {WarpPerCol}, {kM}, {kN}, {kK}, {kP}, {kTM}, {kTN},
              {kTK}, {kTP}>(A, B, C, D);
    return 0;
}}
"""
