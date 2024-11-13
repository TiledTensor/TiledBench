#pragma once

#include "utils/cpp/cutlass/traits_base.cuh"

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_SM80_ENABLED
#endif

template <int N>
DEVICE void wait_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

DEVICE void commit_copy_group() {
#if defined(CP_ASYNC_SM80_ENABLED)
    cute::cp_async_fence();
#endif
}

DEVICE void __copy_async() {
    commit_copy_group();
    wait_group<0>();
}

// Copy a 2d data tile from global memory to shared memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_tile_g2s(const Element* src_data, Element* dst_data,
                          SrcLayout src_layout, DstLayout dst_layout,
                          TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto gtile = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

// Copy a tensor from shared memory to global memory
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_tile_s2g(const Element* src_data, Element* dst_data,
                          SrcLayout src_layout, DstLayout dst_layout,
                          TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto stile = make_tensor(make_smem_ptr(src_data), src_layout);
    auto gtile = make_tensor(make_gmem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(stile);
    auto dst = loader.partition_D(gtile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

/// Copy from FractalTensor.
// TODO(ying): Figure out how to use the same interface for both G2S
// and S2G
// @param ThreadsShape: the shape of the thread block, it should has
// a type of TileShape.
template <typename Element,  //
          const int kThreadsPerRow,
          const int kThreadsPerCol,  //
          typename SrcLayout,
          typename DstLayout,  //
          typename Base = TraitsBase<Element>>
struct G2SCopy2D : public Base {
    using SrcLayout_ = SrcLayout;
    using DstLayout_ = DstLayout;

    // static constexpr int kThreadsPerRow = dim_size<0, ThreadsShape>;
    // static constexpr int kThreadsPerCol = dim_size<1, ThreadsShape>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;
#if defined(CP_ASYNC_ENABLED)
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, Element>;
#endif
    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

  public:
    DEVICE void copy(const Element* src_data, Element* trg_data, int tid) {
        auto gtile = make_tensor(make_gmem_ptr(src_data), SrcLayout_{});
        auto stile = make_tensor(make_smem_ptr(trg_data), DstLayout_{});

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(gtile);
        auto dst = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(_, i, j), dst(_, i, j));
    }

  private:
    TiledCopy tiled_copy_;
};

/// Copy from FractalTensor.
// @param ThreadsShape: the shape of the thread block, it should has
// a type of TileShape.
template <typename Element,  //
          const int kThreadsPerRow,
          const int kThreadsPerCol,  //
          typename SrcLayout,
          typename DstLayout,  //
          typename Base = TraitsBase<Element>>
struct S2GCopy2D : public Base {
    using SrcLayout_ = SrcLayout;
    using DstLayout_ = DstLayout;

    // static constexpr int kThreadsPerRow = dim_size<0, ThreadsShape>;
    // static constexpr int kThreadsPerCol = dim_size<1, ThreadsShape>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;

    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

  public:
    DEVICE void copy(const Element* src_data, Element* trg_data, int tid) {
        auto stile = make_tensor(make_smem_ptr(src_data), SrcLayout_{});
        auto gtile = make_tensor(make_gmem_ptr(trg_data), DstLayout_{});

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(stile);
        auto dst = loader.partition_D(gtile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(_, i, j), dst(_, i, j));
    }

  private:
    TiledCopy tiled_copy_;
};

template <typename Element, typename TiledMma_, typename DstLayout>
struct R2SCopy2D {
    using TiledMma = TiledMma_;
    using Dstlayout_ = DstLayout;
    using CopyAtom = Copy_Atom<DefaultCopy, Element>;

  public:
    template <typename Engine, typename Layout>
    __device__ void copy(cute::Tensor<Engine, Layout> const& acc,
                         Element* dst_data) {
        int tid = threadIdx.x;

        // FIXME(haruhi): This implementation is specifically designed
        // for tcu WMMA and assumes that the ACC value has a
        // floating-point precision. The code converts the ACC value
        // to half-precision.
        auto src_tensor = convert_type<Element>(acc);
        auto dst_tensor = make_tensor(make_smem_ptr(dst_data), DstLayout{});

        auto tiled_copy = make_tiled_copy_C(CopyAtom{}, TiledMma{});
        auto thrd_copy = tiled_copy.get_thread_slice(tid);

        auto src = thrd_copy.retile_S(src_tensor);
        auto dst = thrd_copy.partition_D(dst_tensor);
        cute::copy(tiled_copy, src, dst);
    }

  private:
    template <typename To_type, typename Engine, typename Layout>
    DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(
            *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
                tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
};

template <typename TiledCopy, typename STensor, typename DTensor,
          typename DTensorView>
struct Shm2RegLoad {
  public:
    DEVICE Shm2RegLoad(TiledCopy& copy, const STensor& src, DTensor& dst,
                       DTensorView& dst_view)
        : tiled_copy_(copy), src_(src), dst_(dst), dst_view_(dst_view) {}

    DEVICE void copy(int pos) {
        cute::copy(tiled_copy_, src_(_, _, pos), dst_view_(_, _, pos));
    }

    DEVICE int get_iters() { return size<2>(dst_); }

    DEVICE const auto operator[](int idx) { return dst_(_, _, idx); }

  private:
    TiledCopy& tiled_copy_;
    const STensor& src_;
    DTensor& dst_;
    DTensorView& dst_view_;
};

template <const int kM, const int kN, typename TiledMma>
DEVICE auto get_acc(const TiledMma& tiled_mma) {
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kM>, Int<kN>>{});
    clear(acc);

    return acc;
}

template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rA(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    auto tensor = cute::make_tensor(make_smem_ptr(data), layout);

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);

    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_A(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

// FIXIME(haruhi): the current implementation is for fast experiment,
// it is coupled shared memory layout with the register layout
template <typename Element, typename Layout, typename TiledMma>
DEVICE auto make_s2rB(const Element* data, const Layout& layout,
                      const TiledMma& tiled_mma) {
    int tid = threadIdx.x;

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_B(SmemLoadAtom{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);

    auto tensor = make_tensor(make_smem_ptr(data), layout);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_B(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks
