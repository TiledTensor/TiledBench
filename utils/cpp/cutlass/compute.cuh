#pragma once

#include <cute/algorithm/axpby.hpp>
#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

template <class XEngine, class XLayout>
DEVICE void cute_tanh(cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = tanh(tensor(i));
    }
}

template <class XEngine, class XLayout>
DEVICE void cute_sigmoid(cute::Tensor<XEngine, XLayout>& tensor) {
#pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = 1.0 / (1.0 + exp(-tensor(i)));
    }
}
}  // namespace cutlass_wrapper
}  // namespace benchmarks
