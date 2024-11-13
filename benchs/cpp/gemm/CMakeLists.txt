cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(gemm_bench LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/../../../scripts/cmake")
include(generic)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../../3rd-party")
set(ROOT_DIR "${PROJECT_SOURCE_DIR}/../../../")

include_directories("${THIRD_PARTY_DIR}/cutlass/include")
include_directories("${THIRD_PARTY_DIR}/TiledCUDA/include")

include_directories("${ROOT_DIR}/")
include_directories("${PROJECT_SOURCE_DIR}/cutlass")

add_executable(bench_gemm bench.cu)
target_link_libraries(bench_gemm ${CUDA_CUBLAS_LIBRARIES})
