#include "util.cuh"
#include "utils/cpp/cuda_info.cuh"

#include <cutlass/half.h>

#include <fstream>
#include <iomanip>

//// =============== Test Config=============== ////
static const int kWarpPerRow = 2;
static const int kWarpPerCol = 4;
using WholeShape = GemmShape<4096, 4096, 4096>;
using CtaTileShape = GemmShape<128, 128, 64>;
using WarpLayout = tl::RowMajor<kWarpPerRow, kWarpPerCol>;
static constexpr int kRK = 16;

void run_test(std::ofstream& fout) {
    //// =============== Declaration =============== ////
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    using InType = cutlass::half_t;
    using AccType = float;

    using Config = KeGemmTraits<InType, AccType, WholeShape, CtaTileShape, kRK,
                                WarpLayout>;
    auto tiledcuda_gemm =
        &gemm<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
              typename Config::GIteratorA, typename Config::SIteratorA,
              typename Config::SharedA, typename Config::RegA,
              typename Config::G2SLoaderA, typename Config::S2RLoaderA,
              typename Config::GIteratorB, typename Config::SIteratorB,
              typename Config::SharedB, typename Config::RegB,
              typename Config::G2SLoaderB, typename Config::S2RLoaderB,
              typename Config::GlobalC, typename Config::SharedC,
              typename Config::RegC, typename Config::R2SStorerC,
              typename Config::S2GStorerC>;

    static constexpr int inputs = kTK * (kTN + kTM) * sizeof(InType);
    static constexpr int accumulators = kTM * kTN * sizeof(AccType);
    static constexpr int smem_size =
        inputs > accumulators ? inputs : accumulators;

    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(tiledcuda_gemm,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }
    int block_x = benchmarks::CeilDiv<kM, kTM>;
    int block_y = benchmarks::CeilDiv<kN, kTN>;
    dim3 dim_grid(block_x, block_y, 1);
    dim3 dim_block(Config::kThreads, 1, 1);

    auto cutlass_gemm =
        &cute_gemm<InType, kWarpPerRow, kWarpPerCol, kM, kN, kK, kTM, kTN, kTK>;

    //// =============== Prepare data =============== ////
    // input matrix A
    thrust::host_vector<InType> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());
    thrust::device_vector<InType> d_a = h_a;
    const InType* dA = thrust::raw_pointer_cast(d_a.data());

    // input matrix B
    thrust::host_vector<InType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());
    thrust::device_vector<InType> d_b = h_b;
    const InType* dB = thrust::raw_pointer_cast(d_b.data());

    // output matrix C for cutlass GEMM kernel
    thrust::device_vector<InType> d_c(kM * kN);
    thrust::fill(d_c.begin(), d_c.end(), static_cast<InType>(0.));
    InType* dC = thrust::raw_pointer_cast(d_c.data());

    cutlass_gemm(dA, dB, dC);
    cudaDeviceSynchronize();
    thrust::host_vector<InType> h_c = d_c;

    // tiled cuda gemm kernel
    thrust::device_vector<AccType> d_c2(kM * kN);
    thrust::fill(d_c2.begin(), d_c2.end(), static_cast<AccType>(0.));
    AccType* dC2 = thrust::raw_pointer_cast(d_c2.data());

    tiledcuda_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC2);
    cudaDeviceSynchronize();
    thrust::host_vector<AccType> h_c2 = d_c2;

    bool passed1 = check_results(
        thrust::raw_pointer_cast(h_c.data()) /*cutlass*/,
        thrust::raw_pointer_cast(h_c2.data()) /*tiled cuda*/, kM * kN);

    // cublas
    const __half* dA2 = reinterpret_cast<const __half*>(dA);
    const __half* dB2 = reinterpret_cast<const __half*>(dB);
    thrust::device_vector<__half> d_c3(kM * kN);
    thrust::fill(d_c3.begin(), d_c3.end(), static_cast<__half>(0.));

    cublas_hgemm(kM, kN, kK, dA2, dB2, thrust::raw_pointer_cast(d_c3.data()),
                 false /*timeit*/);
    thrust::host_vector<__half> h_c3 = d_c3;

    bool passed2 = check_results(
        thrust::raw_pointer_cast(h_c3.data()) /*cutlass*/,
        thrust::raw_pointer_cast(h_c2.data()) /*tiled cuda*/, kM * kN);

    if (!(passed1 && passed2)) {
        std::cerr << "Test failed" << std::endl;
        return;
    }

    //// =============== Timing =============== ////
    thrust::fill(d_c.begin(), d_c.end(), static_cast<InType>(0.));
    thrust::fill(d_c2.begin(), d_c2.end(), static_cast<AccType>(0.));
    thrust::fill(d_c3.begin(), d_c3.end(), static_cast<__half>(0.));
    // cublas
    float cublas_time =
        cublas_hgemm(kM, kN, kK, dA2, dB2,
                     thrust::raw_pointer_cast(d_c3.data()), true /*timeit*/);

    const int warm_up = 5;
    const int iters = 20;
    for (int i = 0; i < warm_up; ++i) {
        cutlass_gemm(dA, dB, dC);
        tiledcuda_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC2);
    }
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        cutlass_gemm(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    float cutlass_time = timer.stop() / iters;

    timer.start();
    for (int i = 0; i < iters; ++i) {
        tiledcuda_gemm<<<dim_grid, dim_block, smem_size>>>(dA, dB, dC2);
    }
    cudaDeviceSynchronize();
    float tiledcuda_time = timer.stop() / iters;

    fout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM << ", "
         << kTN << ", " << kTK << "]\t" << kRK << "\t[" << kWarpPerRow << ", "
         << kWarpPerCol << "]\t" << cublas_time << "\t" << cutlass_time << " ("
         << std::setprecision(2) << cutlass_time / cublas_time << ")"
         << "\t" << std::setprecision(4) << tiledcuda_time << " ("
         << std::setprecision(2) << tiledcuda_time / cublas_time << ")"
         << std::endl;
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    auto dev_name = benchmarks::get_device_name();
    std::stringstream file_name;
    file_name << "figures/bench_" << dev_name << "_gemm.tsv";
    fout.open(file_name.str(), std::ios::out);

    fout << "[M, N, K]\t[kTM, kTN, kTK]\tkRK\tWarp Layout\t"
            "cuBLAS(ms)\tcutlass(ms)\tTiledCUDA(ms)"
         << std::endl;

    run_test(fout);

    return 0;
}
