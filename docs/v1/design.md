# v1: Per-expert C++ / cuBLAS baseline

## Purpose and contract

v1 is the first native execution baseline. It implements the same FP32
post-routing GEMM and packed layout as [v0](../v0/design.md), including its
versioned little-endian file contract. It uses one CUDA stream and one
`cublasSgemm` call per active expert. It does not collect performance data.

## Interfaces and execution

Build with CMake and run `build/moe_v1 INPUT.bin OUTPUT.bin`. The process
validates the input before allocating CUDA resources, uploads X and W,
submits expert GEMMs, synchronizes the stream, downloads Y, and writes the
output. Offsets stay on the CPU; no routing metadata transfer is needed.
Zero-token inputs produce an empty output without initializing CUDA.

`run_experts` in `src/v1/baseline.hpp` accepts device pointers for X/W/Y,
host offsets, K/N, a caller-owned cuBLAS handle, and a stream. Allocation,
file I/O, and synchronization remain outside this function. The caller
provides validated dimensions and disjoint appropriately sized buffers;
the function enqueues work asynchronously and binds the handle to the
stream. Host pointer mode and default FP32 math are selected explicitly.
Later grouped or tiled implementations can reuse the same packed arrays.

## Row-major mapping

For an expert with M tokens, row-major `Y[M,N] = X[M,K] W[K,N]` is
equivalent to column-major `Y^T[N,M] = W^T[N,K] X^T[K,M]` on the same bytes.
No physical transpose or padding is needed.

| cuBLAS argument | Value |
| --- | --- |
| transa, transb | CUBLAS_OP_N, CUBLAS_OP_N |
| m, n, k | N, M, K |
| A, lda | W + e*K*N, N |
| B, ldb | X + offsets[e]*K, K |
| C, ldc | Y + offsets[e]*N, N |
| alpha, beta | 1.0f, 0.0f |

Empty experts issue no GEMM. The loop contains no per-expert synchronization.
The implementation uses `CUBLAS_DEFAULT_MATH` and FP32 SGEMM; v0 disables
TF32. FP16 and alternate math modes are outside this version.

## Build, ownership and failures

CMake enables C++17 only and links `CUDA::cublas` and `CUDA::cudart` from
the system CUDA Toolkit. No nvcc compilation is needed yet. Python's conda
CUDA runtime is separate from this development toolkit.

On the local WSL machine, the system runtime otherwise resolves a conflicting
Linux `libcuda.so.1`. Set `LD_LIBRARY_PATH` to prioritize `/usr/lib/wsl/lib`
as documented in the README. This remains a launch-environment setting,
not a hardcoded driver path or a system-library modification.

Device buffers, stream, and handle use RAII. Normal completion explicitly
checks synchronization and cleanup calls; destructors provide nonthrowing
cleanup on errors and report cleanup failures to stderr. CUDA/cuBLAS errors,
bad files and allocation failures result in a diagnostic and nonzero exit.
The reader checks integer multiplication overflow, int32 dimension limits,
exact payload size, offsets, and finite values. Big-endian hosts are rejected.

## Acceptance and limitations

Use the shared correctness runner and pytest suite described in v0.
Rectangular K != N and non-aligned dimensions exercise the leading-dimension
mapping. Both implementations must meet the stated tolerance against a
small CPU FP64 reference. Malformed inputs must fail clearly.

There is no timing or profiling. This baseline preserves cuBLAS's internal
GEMM scheduling; future cross-expert tile scheduling requires a separate
kernel or configurable GEMM implementation. No speedup is claimed.
