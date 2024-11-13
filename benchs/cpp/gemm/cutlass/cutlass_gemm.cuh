#pragma once

#include "utils/cpp/cuda_utils.cuh"
#include "utils/cpp/cutlass/copy.cuh"
#include "utils/cpp/cutlass/layout.cuh"
#include "utils/cpp/cutlass/traits_base.cuh"

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

/// Copy from FractalTensor.
template <typename Element_,                             //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK,   //
          typename Base = TraitsBase<Element_>>
struct KeGemmTraits : public Base {
    using Element = Element_;
    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    static constexpr int kNumPerAccess = Base::kNumPerAccess;

    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    using SmemLayoutAtom = cute::Layout<Shape<_8, _32>, Stride<_32, _1>>;
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));

    // using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;
    using LoadA_G2S = G2SCopy2D<Element, kThreadsPerRow, kThreadsPerCol,
                                RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    // NOTE: the input matrix B: [kK, kN] is physically laid out in
    // a column major format, that is, the K dimension is contiguous
    // in memory. However, a physically column major matrix can be
    // viewed as a row major matrix with a transpose. Therefore, we
    // can use the `RowMajor` layout here.
    //   using ThreadShapeB = TileShape<kThreadsPerRow,
    //   kThreadsPerCol>;
    using LoadB_G2S = G2SCopy2D<Element, kThreadsPerRow, kThreadsPerCol,
                                RowMajor<kTN, kTK, kK>, SmemLayoutB>;

    // TODO(ying): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    using StoreC_R2S = R2SCopy2D<Element, TiledMma, RowMajor<kTM, kTN>>;

    using StoreC_S2G =
        S2GCopy2D<Element, kThreadsPerRow, kThreadsPerCol,
                  RowMajor<kTM, kTN> /*shared memory layout*/,
                  RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

/// GEMM kernel using cutlass3 APIs
/// Copy from FractalTensor.
template <typename Element, const int kM, const int kN, const int kK,
          const int kTM, const int kTN, const int kTK, typename KeTraits>
__global__ void KeCuteGemm(const Element* dA, const Element* dB, Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan
    typename KeTraits::StoreC_S2G gC;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc);  // compute using tcu's wmma instruction
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    gC.copy(shm, gC_ptr,
            tid);  // store shared memory tile to global memory
}

}  // namespace cutlass_wrapper
}  // namespace benchmarks

template <typename Element,                              //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK>
void cute_gemm(const Element* dA, const Element* dB, Element* dC) {
    static_assert(kTM % kWarpPerRow == 0,
                  "the M dimension of the CTA tile should be "
                  "divisible by the "
                  "number of warps along that that dimension.");
    static_assert(kTN % kWarpPerCol == 0,
                  "the N dimension of the CTA tile should be "
                  "divisible by the "
                  "number of warps along that that dimension.");

    using KeTraits = benchmarks::cutlass_wrapper::KeGemmTraits<
        Element, kWarpPerRow, kWarpPerCol, kM, kN, kK, kTM, kTN, kTK>;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

    auto kernel =
        &benchmarks::cutlass_wrapper::KeCuteGemm<Element, kM, kN, kK, kTM, kTN,
                                                 kTK, KeTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    const int block_m = (kM + kTM - 1) / kTM;
    const int block_n = (kN + kTN - 1) / kTN;

    const int kThreads = KeTraits::kThreads;

    dim3 gridDim(block_m, block_n);
    dim3 blockDim(kThreads, 1, 1);
    kernel<<<gridDim, blockDim, smem_size>>>(dA, dB, dC);
}
