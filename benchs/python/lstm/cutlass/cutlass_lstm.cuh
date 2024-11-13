#pragma once

#include "utils/cpp/cuda_utils.cuh"
#include "utils/cpp/cutlass/compute.cuh"
#include "utils/cpp/cutlass/copy.cuh"
#include "utils/cpp/cutlass/traits_base.cuh"

#include <cute/tensor.hpp>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

template <typename Element_,                             //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK,   //
          typename Base = TraitsBase<Element_>>
struct LstmTraits : public Base {
    using Element = Element_;

    static_assert(kTM % kWarpPerRow == 0,
                  "the M dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");
    static_assert(kTN % kWarpPerCol == 0,
                  "the N dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");

    // declare global to shared memory copy layout.
    using GmemLayoutA = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutB = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutC = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutD = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutE = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kN>, _1>>;

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,  // for ampere
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>, _16>>;

    static constexpr int kThreads = size(TiledMma{});
    static_assert(kThreads == kWarpPerRow * kWarpPerCol * 32);

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<_8, Int<4 * kNumPerAccess>>,
                                   Stride<Int<4 * kNumPerAccess>, _1>>{}));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));
    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutE =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif

    using TiledCopyG2S = decltype(make_tiled_copy(
        CopyInstG2S{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    using TiledCopyS2G = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    using StoreE_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutE>;
};

template <typename Element, const int kM, const int kN, const int kK,
          const int kTM, const int kTN, const int kTK, typename KeTraits>
__global__ void lstm_gate_kernel(const Element* ws, const Element* us,
                                 const Element* xs, const Element* hs,
                                 Element* ts) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Advance to the global data tile to the current CTA.
    Element* gxs_ptr = const_cast<Element*>(xs) + blockIdx.y * kK * kTN;
    Element* ghs_ptr = const_cast<Element*>(hs) + blockIdx.y * kK * kTN;
    Element* gws_ptr = const_cast<Element*>(ws) + blockIdx.x * kK * kTM;
    Element* gus_ptr = const_cast<Element*>(us) + blockIdx.x * kK * kTM;
    Element* gts_ptr = ts + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sws_ptr = shm;
    Element* sxs_ptr = shm + kTM * kTK;
    Element* sus_ptr = shm + kTM * kTK + kTK * kTN;
    Element* shs_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;
    Element* sts_ptr = shm;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy;

    auto rws = make_s2rA(sws_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rxs = make_s2rB(sxs_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto rus = make_s2rA(sus_ptr, typename KeTraits::SmemLayoutC{}, mma);
    auto rhs = make_s2rB(shs_ptr, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sts;  // declare register to shared store

    for (int k = 0; k < kK; k += kTK) {
        copy_tile_g2s(gws_ptr, sws_ptr, typename KeTraits::GmemLayoutA{},
                      typename KeTraits::SmemLayoutA{}, tiled_copy);
        copy_tile_g2s(gxs_ptr, sxs_ptr, typename KeTraits::GmemLayoutB{},
                      typename KeTraits::SmemLayoutB{}, tiled_copy);
        copy_tile_g2s(gus_ptr, sus_ptr, typename KeTraits::GmemLayoutC{},
                      typename KeTraits::SmemLayoutC{}, tiled_copy);
        copy_tile_g2s(ghs_ptr, shs_ptr, typename KeTraits::GmemLayoutD{},
                      typename KeTraits::SmemLayoutD{}, tiled_copy);

        __copy_async();
        __syncthreads();

        for (int i = 0; i < rws.get_iters(); i++) {
            rws.copy(i);
            rxs.copy(i);
            gemm(mma, rws[i], rxs[i], acc1);
        }

        for (int i = 0; i < rus.get_iters(); i++) {
            rus.copy(i);
            rhs.copy(i);
            gemm(mma, rus[i], rhs[i], acc2);
        }

        __syncthreads();
        gws_ptr += kTK;
        gxs_ptr += kTK;
        gus_ptr += kTK;
        ghs_ptr += kTK;
    }

    __syncthreads();
    cute::axpby(1.0, acc1, 1.0, acc2);

    __syncthreads();
    if (current_block_x < total_block_x * 3 / 4) {
        cute_sigmoid(acc2);
    } else {
        cute_tanh(acc2);
    }
    __syncthreads();

    sts.copy(acc2, shm);

    __syncthreads();

    copy_tile_s2g(sts_ptr, gts_ptr, typename KeTraits::SmemLayoutE{},
                  typename KeTraits::GmemLayoutE{},
                  typename KeTraits::TiledCopyS2G{});
}

template <typename Element>
__global__ void lstm_element_wise(const Element* i, const Element* f,
                                  const Element* o, const Element* c_candidate,
                                  const Element* c, Element* c_out,
                                  Element* h_out, const int block_size,
                                  int size) {
    int index = blockIdx.x * block_size + threadIdx.x;
    if (index < size) {
        // TODO: Loading data into shared memory and computing, versus
        // computing directly in global memory, does not seem to make a
        // difference. This seems to require further optimization, such as
        // reconsidering redistributing data to different threads and performing
        // vectorized loading and storing.

        // This is a very naive kernel that loads data into shared memory and
        // then performs computations. It has been temporarily commented out.

        c_out[index] = f[index] * c[index] + i[index] * c_candidate[index];

        __syncthreads();

        h_out[index] = o[index] * tanh(c_out[index]);
    }
}

template <typename Element,                              //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK>
void lstm_gate(const Element* w, const Element* x, const Element* u,
               const Element* h, Element* t) {
    using KeTraits = LstmTraits<Element, kWarpPerRow, kWarpPerCol, kM, kN, kK,
                                kTM, kTN, kTK>;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

    auto kernel =
        &lstm_gate_kernel<Element, kM, kN, kK, kTM, kTN, kTK, KeTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    const int block_m = (kM + kTM - 1) / kTM;
    const int block_n = (kN + kTN - 1) / kTN;

    const int kThreads = KeTraits::kThreads;

    dim3 gridDim(block_m, block_n, 1);
    dim3 blockDim(kThreads, 1, 1);

    kernel<<<gridDim, blockDim, smem_size>>>(w, u, x, h, t);
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks

template <typename Element,                              //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK>
void cute_lstm_cell(const Element* w, const Element* x, const Element* u,
                    const Element* c, const Element* h, Element* c_out,
                    Element* h_out) {
    static const int M = kM / 4;
    static const int N = kN;

    // Cuda malloc for output
    Element* t;
    benchmarks::CudaCheck(cudaMalloc(&t, kM * kN * sizeof(Element)));

    benchmarks::cutlass_wrapper::lstm_gate<Element, kWarpPerRow, kWarpPerCol,
                                           kM, kN, kK, kTM, kTN, kTK>(w, x, u,
                                                                      h, t);

    const Element* i = t;
    const Element* f = t + M * N;
    const Element* o = t + 2 * M * N;
    const Element* c_candidate = t + 3 * M * N;

    auto element_wise =
        &benchmarks::cutlass_wrapper::lstm_element_wise<Element>;

    /*
    TODO: Use `kMaxThreads` will case a runtime error:
    ```
    RuntimeError: CUDA error: invalid configuration argument
    CUDA kernel errors might be asynchronously reported at some other API call,
    so the stacktrace below might be incorrect. For debugging consider passing
    CUDA_LAUNCH_BLOCKING=1. Compile with `TORCH_USE_CUDA_DSA` to enable
    device-side assertions.
    ```
    */
    // int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);
    int size = M * N;
    const int block_threads = 512;
    int block_size = (size + block_threads - 1) / block_threads;
    dim3 element_wise_grid_dim(block_size, 1, 1);
    dim3 element_wise_block_dim(block_threads, 1, 1);

    element_wise<<<element_wise_grid_dim, element_wise_block_dim>>>(
        i, f, o, c_candidate, c, c_out, h_out, block_threads, size);

    benchmarks::CudaCheck(cudaFree(t));
}
