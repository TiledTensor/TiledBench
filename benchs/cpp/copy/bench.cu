#include "cutlass/cutlass_copy.cuh"
#include "tiledcuda/tiledcuda_copy.cuh"
#include "utils/cpp/cuda_info.cuh"
#include "utils/cpp/cuda_timer.cuh"

#include <cutlass/half.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cfloat>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

float rand_float(float a = 1e-4, float b = 1e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

void run_test(std::ofstream& fout) {
    using Element = cutlass::half_t;
    using InType = __half;
    using AccType = float;

    static constexpr int kM = 4096;
    static constexpr int kN = 4096;
    static constexpr int kK = 2048;

    static constexpr int kTM = 64;
    static constexpr int kTN = 32;
    static constexpr int kTK = 32;

    static constexpr int kWarpPerRow = 2;
    static constexpr int kWarpPerCol = 2;

    thrust::host_vector<Element> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<Element>(rand_float());

    thrust::host_vector<Element> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::host_vector<AccType> h_c2(kM * kN);
    thrust::fill(h_c2.begin(), h_c2.end(), 0.);

    thrust::device_vector<Element> d_a = h_a;
    thrust::device_vector<Element> d_b = h_b;
    thrust::device_vector<Element> d_c = h_c;
    thrust::device_vector<AccType> d_c2 = h_c2;

    const Element* dA = thrust::raw_pointer_cast(d_a.data());
    const Element* dB = thrust::raw_pointer_cast(d_b.data());
    Element* dC = thrust::raw_pointer_cast(d_c.data());

    const InType* dA2 = reinterpret_cast<const InType*>(dA);
    const InType* dB2 = reinterpret_cast<const InType*>(dB);
    AccType* dC2 = thrust::raw_pointer_cast(d_c2.data());

    auto cute_shared_copy_kernel =
        &cute_shared_copy<Element, kWarpPerRow, kWarpPerCol, kM, kN, kK, kTM,
                          kTN, kTK>;

    auto cute_copy_kernel = &cute_copy<Element, kWarpPerRow, kWarpPerCol, kM,
                                       kN, kK, kTM, kTN, kTK>;

    auto tiledcuda_shared_copy_kernel =
        &tiledcuda_shared_copy<InType, AccType, kM, kN, kK, kTM, kTN, kTK,
                               kWarpPerRow, kWarpPerCol>;

    auto tiledcuda_copy_kernel =
        &tiledcuda_copy<InType, AccType, kM, kN, kK, kTM, kTN, kTK, kWarpPerRow,
                        kWarpPerCol>;

    const int warm_up = 5;
    const int iters = 20;

    benchmarks::CudaTimer timer;

    for (int i = 0; i < warm_up; ++i) {
        cute_shared_copy_kernel(dA, dB, dC);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        cute_shared_copy_kernel(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    float cutlass_time = timer.stop() / iters;

    for (int i = 0; i < warm_up; ++i) {
        cute_copy_kernel(dA, dB, dC);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        cute_copy_kernel(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    float cutlass_time2 = timer.stop() / iters;

    for (int i = 0; i < warm_up; ++i) {
        tiledcuda_shared_copy_kernel(dA2, dB2, dC2);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        tiledcuda_shared_copy_kernel(dA2, dB2, dC2);
    }
    cudaDeviceSynchronize();
    float tiledcuda_time = timer.stop() / iters;

    for (int i = 0; i < warm_up; ++i) {
        tiledcuda_copy_kernel(dA2, dB2, dC2);
    }

    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < iters; ++i) {
        tiledcuda_copy_kernel(dA2, dB2, dC2);
    }

    cudaDeviceSynchronize();
    float tiledcuda_time2 = timer.stop() / iters;

    std::cout << "Whole\t" << "[" << kM << ", " << kN << ", " << kK << "]\t["
              << kTM << ", " << kTN << ", " << kTK << "]\t[" << kWarpPerRow
              << ", " << kWarpPerCol << "]\t" << cutlass_time2 << "\t"
              << tiledcuda_time2 << "\t" << tiledcuda_time2 / cutlass_time2
              << std::endl;

    std::cout << "G2S\t" << "[" << kM << ", " << kN << ", " << kK << "]\t["
              << kTM << ", " << kTN << ", " << kTK << "]\t[" << kWarpPerRow
              << ", " << kWarpPerCol << "]\t" << cutlass_time << "\t"
              << tiledcuda_time << "\t" << tiledcuda_time / cutlass_time
              << std::endl;

    fout << "Whole\t" << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM
         << ", " << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", "
         << kWarpPerCol << "]\t" << cutlass_time2 << "\t" << tiledcuda_time2
         << "\t" << tiledcuda_time2 / cutlass_time2 << std::endl;

    fout << "G2S\t" << "[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM
         << ", " << kTN << ", " << kTK << "]\t[" << kWarpPerRow << ", "
         << kWarpPerCol << "]\t" << cutlass_time << "\t" << tiledcuda_time
         << "\t" << tiledcuda_time / cutlass_time << std::endl;
}

int main() {
    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    auto dev_name = benchmarks::get_device_name();
    std::stringstream file_name;
    file_name << "bench_" << dev_name << "_copy.tsv";
    fout.open(file_name.str(), std::ios::out);

    fout << "Copy Type\t"
         << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, "
            "kWarpPerCol]\tCutlassTime(ms)\tTiledCUDATime(ms)\tRatio"
         << std::endl;

    std::cout << "Copy Type\t"
              << "[M, N, K]\t[kTM, kTN, kTK]\t[kWarpPerRow, "
                 "kWarpPerCol]\tCutlassTime(ms)\tTiledCUDATime(ms)\tRatio"
              << std::endl;

    run_test(fout);

    return 0;
}
