cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(cublas_gemm LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/../../../../../scripts/cmake")
include(generic)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../../../../3rd-party")
include_directories("${THIRD_PARTY_DIR}/cutlass/include")
include_directories("${THIRD_PARTY_DIR}/TiledCUDA/include")

cuda_add_library(cublas_gemm SHARED bind.cu)
target_link_libraries(cublas_gemm ${TORCH_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
