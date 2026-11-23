# v0: PyTorch correctness reference

## Purpose and computation

v0 defines the numerical contract for a top-1, post-routing MoE expert
GEMM. It computes `Y_e = X_e @ W_e` in FP32, without bias or activation.
The correctness reference and benchmark share the expert GEMM loop. Routing,
packing, and restoring the original token order are outside computation.

`X[T,K]`, `W[E,K,N]`, and `Y[T,N]` are contiguous row-major FP32 arrays.
`offsets[E+1]` is int64, starts at zero, ends at T, and is nondecreasing.
Expert e owns rows `[offsets[e], offsets[e+1])`. Empty experts are skipped.
T may be zero; E, K, and N must be positive. Dimensions fit signed 32-bit
cuBLAS arguments; payload sizes are checked before allocation.

## Input generation and execution

`src/moe/workloads.py` generates deterministic NumPy inputs from a seed.
Balanced routing divides T as evenly as possible; uniform routing samples
expert IDs independently; hotspot routing assigns exactly ceil(T/2) tokens
to expert zero and distributes the remainder among other experts. Explicit
counts must be nonnegative, have length E, and sum to T. Inputs are already
packed. Input features are standard normal; weights are scaled by 1/sqrt(K).

`src/moe/reference.py` transfers arrays to the selected PyTorch device,
disables TF32, and evaluates each active expert under inference mode.
The default device is CUDA. CPU execution is available for portability.
A separate CPU FP64 reference uses per-token matrix-vector products,
independently of expert slicing, for small validation cases.

## File interface (version 1)

All integers and array elements are little-endian; no padding is present.
The Python implementation lives in `src/moe/format.py`.

| File | Ordered fields |
| --- | --- |
| Input | 8-byte magic `MOEIN001`, uint32 version=1, uint64 T,E,K,N, int64 offsets[E+1], float32 X[T,K], float32 W[E,K,N] |
| Output | 8-byte magic `MOEOUT01`, uint32 version=1, uint64 T,N, float32 Y[T,N] |

Readers reject unsupported magic/version, invalid dimensions or offsets,
truncated or surplus bytes, and non-finite arrays. File size is checked
before loading payloads. Writers enforce the same contract. Generated
files belong under ignored `artifacts/`, not source control.

## Validation and limits

`scripts/check_correctness.py` generates fixtures, runs v0 and the v1
executable, reloads outputs, and checks `atol=1e-4, rtol=1e-4`. It also
compares both outputs against the CPU FP64 reference and reports maximum
absolute error. Tests cover uneven counts, empty experts, zero tokens,
single active experts, T<E, rectangular and non-aligned GEMMs, and malformed
files. Non-finite outputs fail validation. The CUDA benchmark measures the
expert loop with resident GPU buffers. The [v0 experiment report](report.md)
includes the configuration and results. Full MLPs, top-k routing
and unpermutation are deferred.
