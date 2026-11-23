#pragma once

#include <cstdint>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/** Throw an exception naming the operation if a CUDA call failed. */
void check_cuda(cudaError_t status, const char* operation);
/** Throw an exception naming the operation if a cuBLAS call failed. */
void check_cublas(cublasStatus_t status, const char* operation);

/** Enqueue expert GEMMs on stream; the caller owns validated buffers and resources.
 * X[T,K], W[E,K,N], and Y[T,N] are row-major; offsets delimit expert rows.
 * This function binds the handle to stream but does not synchronize it.
 */
void run_experts(const float* x, const float* w, float* y,
                 const std::vector<std::int64_t>& offsets, int k, int n,
                 cublasHandle_t handle, cudaStream_t stream);
