#include "cutlass/cutlass_gemm.cuh"
#include "tiledcuda/tiledcuda_gemm.cuh"
#include "util.cuh"
#include "utils/cpp/cuda_info.cuh"

#include <cutlass/half.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

template <typename Element,                             //
          const int kM, const int kN, const int kK,     //
          const int kTM, const int kTN, const int kTK,  //
          const int kWarpPerRow, const int kWarpPerCol>
void run_test(std::ofstream& fout) {
    using InType = __half;
    using AccType = float;

    thrust::host_vector<Element> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<Element> d_c = h_c;

    const Element* dA = thrust::raw_pointer_cast(d_a.data());
    const Element* dB = thrust::raw_pointer_cast(d_b.data());
    Element* dC = thrust::raw_pointer_cast(d_c.data());

    auto kernel = &cute_gemm<Element, kWarpPerRow, kWarpPerCol, kM, kN, kK, kTM,
                             kTN, kTK>;

    kernel(dA, dB, dC);

    cudaDeviceSynchronize();

    h_c = d_c;

    // Check results
    thrust::device_vector<__half> d_c2(kM * kN);
    thrust::fill(d_c2.begin(), d_c2.end(), 0.);

    const __half* dA2 = reinterpret_cast<const __half*>(dA);
    const __half* dB2 = reinterpret_cast<const __half*>(dB);

    cublas_hgemm(kM, kN, kK, dA2, dB2, thrust::raw_pointer_cast(d_c2.data()),
                 false /*timeit*/);
    thrust::host_vector<__half> h_c2 = d_c2;

    bool passed = check_results(thrust::raw_pointer_cast(h_c.data()),
                                thrust::raw_pointer_cast(h_c2.data()), kM * kN);

    if (passed) {
        const int warm_up = 5;
        const int iters = 20;

        for (int i = 0; i < warm_up; ++i) {
            kernel(dA, dB, dC);
        }

        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iters; ++i) {
            kernel(dA, dB, dC);
        }
        cudaDeviceSynchronize();
        float cutlass_time = timer.stop() / iters;

        float cublas_time = cublas_hgemm(kM, kN, kK, dA2, dB2,
                                         thrust::raw_pointer_cast(d_c2.data()),
                                         true /*timeit*/);
        std::cout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM
                  << ", " << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", "
                  << kWarpPerCol << "]\t" << cublas_time << "(ms)\t"
                  << cutlass_time << "(ms)\t" << cutlass_time / cublas_time
                  << std::endl;

        fout << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM << ", "
             << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", "
             << kWarpPerCol << "]\t" << cublas_time << "\t" << cutlass_time
             << "\t" << cutlass_time / cublas_time << std::endl;
    } else {
        std::cerr << "Test failed" << std::endl;
    }
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    auto dev_name = benchmarks::get_device_name();
    std::stringstream file_name;
    file_name << "bench_" << dev_name << "_gemm.tsv";
    fout.open(file_name.str(), std::ios::out);

    fout << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, kWarpPerCol]\t"
            "CublasTime(ms)\tCutlassTime(ms)\tRatio"
         << std::endl;

    std::cout << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, "
                 "kWarpPerCol]\tCublasTime(ms)\tCutlassTime(ms)\tRatio"
              << std::endl;

    run_test<cutlass::half_t, 4096, 4096, 32, 64, 32, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 32, 64, 64, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 32, 64, 256, 32, 2, 2>(fout);

    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 32, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 64, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 256, 32, 2, 2>(fout);

    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 256, 64, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 128, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 64, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 128, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 32, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 32, 64, 32, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 256, 128, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 256, 128, 128, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 256, 128, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 128, 128, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 128, 64, 64, 2, 2>(fout);
    run_test<cutlass::half_t, 4096, 4096, 2048, 64, 128, 64, 2, 2>(fout);

    return 0;
}
