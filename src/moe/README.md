# Python reference and workload tools

This package generates packed MoE inputs and computes correctness references.
Each expert processes its own token rows: `Y_e = X_e @ W_e`.

| File | Responsibility |
| --- | --- |
| `workloads.py` | Generate deterministic balanced, uniform, hotspot, or explicit expert loads. |
| `format.py` | Validate tensors and read/write the binary files shared with C++. |
| `reference.py` | Compute the PyTorch FP32 reference and a small CPU FP64 cross-check. |
| `__main__.py` | Expose the `generate` and `reference` commands. |

`X[T,K]`, `W[E,K,N]`, and `Y[T,N]` are row-major FP32 arrays.
`offsets[E+1]` identifies each expert's contiguous token rows. T is the token
count, E the expert count, K the input width, and N the output width.

Run from the repository root with the `ML` environment activated:

```bash
mkdir -p artifacts
PYTHONPATH=src python -m moe generate artifacts/input.bin --tokens 9 --experts 3 --k 5 --n 7 --counts 4 0 5
PYTHONPATH=src python -m moe reference artifacts/input.bin artifacts/v0.bin
```

See the [v0 design](../../docs/v0/design.md) for the file format and
[project README](../../README.md) for setup and correctness checks.
