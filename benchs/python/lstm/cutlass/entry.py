entry = """#include "../cutlass_lstm.cuh"

extern "C" int kernel_entry(const __half* dW, const __half* dX, const __half* dU,
const __half* dC, const __half* dH, __half* dCO, __half* dHO) {{
    using DType = cutlass::half_t;

    auto* W = reinterpret_cast<const DType*>(dW);
    auto* X = reinterpret_cast<const DType*>(dX);
    auto* U = reinterpret_cast<const DType*>(dU);
    auto* C = reinterpret_cast<const DType*>(dC);
    auto* H = reinterpret_cast<const DType*>(dH);
    auto* CO = reinterpret_cast<DType*>(dCO);
    auto* HO = reinterpret_cast<DType*>(dHO);

    cute_lstm_cell<DType, {WarpPerRow}, {WarpPerCol}, {kM}, {kN}, {kK}, {kTM}, {kTN},
              {kTK}>(W, X, U, C, H, CO, HO);
    return 0;
}}
"""
