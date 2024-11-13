#pragma once

#include <cute/layout.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int row, const int col, const int stride = col>
using RowMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<Int<stride>, _1>>;

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int row, const int col, const int stride = row>
using ColMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<_1, Int<stride>>>;

template <typename Layout>
static constexpr size_t num_rows = cute::size<0>(Layout{});

template <typename Layout> /*  */
static constexpr size_t num_cols = cute::size<1>(Layout{});

}  // namespace cutlass_wrapper
}  // namespace benchmarks