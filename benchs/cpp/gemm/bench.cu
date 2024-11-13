#include "cutlass/cutlass_gemm.cuh"
#include "util.cuh"

#include <cutlass/half.h>

template <typename Element,                             //
          const int kM, const int kN, const int kK,     //
          const int kTM, const int kTN, const int kTK,  //
          const int kWarpPerRow, const int kWarpPerCol>
void run_test() {
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
}

int main() { run_test<cutlass::half_t, 4096, 4096, 32, 64, 32, 32, 2, 2>(); }