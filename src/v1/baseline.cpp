#include "baseline.hpp"

#include <stdexcept>
#include <string>

/** Convert a failed CUDA status into a readable exception. */
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

/** Convert a failed cuBLAS status into an exception with its status code. */
void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}

/** Submit one FP32 GEMM per nonempty expert without waiting for completion. */
void run_experts(const float* x, const float* w, float* y,
                 const std::vector<std::int64_t>& offsets, int k, int n,
                 cublasHandle_t handle, cudaStream_t stream) {
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    check_cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    check_cublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    // beta=0 overwrites Y; its device buffer does not need initialization.
    const float alpha = 1.0f, beta = 0.0f;
    for (std::size_t e = 0; e + 1 < offsets.size(); ++e) {
        const int m = static_cast<int>(offsets[e + 1] - offsets[e]);
        if (m == 0) continue;
        // cuBLAS sees column-major Y^T = W^T X^T on the same row-major bytes.
        check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, w + e * k * n, n,
                                x + static_cast<std::size_t>(offsets[e]) * k, k,
                                &beta, y + static_cast<std::size_t>(offsets[e]) * n, n),
                     "cublasSgemm");
    }
}
