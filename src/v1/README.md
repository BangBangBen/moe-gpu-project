# C++ / cuBLAS baseline

v1 computes `Y_e = X_e @ W_e` for each active expert, using FP32 SGEMM on a
single CUDA stream. Empty experts are skipped. No performance data is collected.

| File | Responsibility |
| --- | --- |
| `baseline.hpp` | Declare error helpers and the asynchronous expert computation interface. |
| `baseline.cpp` | Submit one cuBLAS GEMM per active expert. |
| `main.cpp` | Validate files, manage CUDA resources, transfer data, and write results. |

The computation function receives already allocated device buffers and host
offsets; the caller owns memory and synchronization. Row-major inputs are
interpreted as transposed column-major matrices, so no transpose copy is needed.

Build and run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
build/moe_v1 artifacts/input.bin artifacts/v1.bin
```

Generate the input with the [Python tools](../moe/README.md).
See the [v1 design](../../docs/v1/design.md) for cuBLAS argument mapping and
the [project README](../../README.md) for dependencies, local WSL configuration,
and correctness checks.
