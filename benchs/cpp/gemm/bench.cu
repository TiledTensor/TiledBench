#include "cutlass/cutlass_gemm.cuh"

template <typename Element,                             //
          const int kM, const int kN, const int kK,     //
          const int kTM, const int kTN, const int kTK,  //
          const int kWarpPerRow, const int kWarpPerCol>
void run_test() {}

int main() { printf("Hello World\n"); }