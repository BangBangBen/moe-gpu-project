# MoE GPU Project

Post-routing Mixture-of-Experts GEMM baselines for CMU 15-618.
The current versions implement FP32 `Y_e = X_e @ W_e` with packed top-1
expert inputs. See the [v0 experiment report](docs/v0/report.md) for the
configuration, results, and reproduction commands.

## Environment

On a Linux x86-64 machine with a CUDA-capable NVIDIA GPU, install a compatible
NVIDIA driver first. Create the Python environment using the declared channels
and dependencies:

```bash
conda env create -f environment.yml
conda activate ML
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

To choose a different environment name, use
`conda env create -n moe-dev -f environment.yml`. An existing ML environment
can run the code if it already satisfies these dependencies; creating a fresh
environment is preferable when reproducing the declared setup.

`environment.yml` pins Python 3.11, PyTorch 2.5.1 and its CUDA runtime 12.4,
and declares NumPy 2.0, pytest 8, and Matplotlib 3.10. It is a direct-dependency specification,
not a complete transitive lockfile. PyTorch and NumPy provide the runtime;
pytest is used for correctness tests.

**C++ prerequisites are separate:** install a C++17 compiler, CMake >= 3.18,
and a system CUDA Toolkit including development headers and cuBLAS/cudart
libraries. The conda CUDA runtime supports PyTorch; it does not replace the
system development toolkit used by CMake. Both use the installed NVIDIA
driver, and do not need identical runtime versions for this file-based
workflow. No Python/C++ extension or nvcc compilation is used in v1.

## Build v1

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If CUDA is installed outside the default search paths, add
`-DCUDAToolkit_ROOT=/path/to/cuda` to the configure command. Build logs identify
the compiler and Toolkit used. The local development setup has an RTX 2080
SUPER, PyTorch 2.5.1 (CUDA 12.4), and system CUDA Toolkit 12.0.

On WSL, if Linux NVIDIA driver libraries shadow the Windows-provided CUDA
driver and v1 reports `no CUDA-capable device is detected`, prioritize the
WSL driver in the shell before running the checks or executable:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This is required on the current local machine. It does not change system
libraries and is not needed on a normal native Linux installation.

## Correctness checks

From the repository root, with ML activated and v1 built:

```bash
python scripts/check_correctness.py --binary build/moe_v1
python -m pytest -q --v1-binary build/moe_v1
```

The runner generates deterministic inputs, evaluates the PyTorch v0 reference,
executes C++ v1, and compares both with a small per-token CPU FP64 reference.
It uses `atol=1e-4, rtol=1e-4` and reports maximum absolute error, with no
performance measurements. Fixtures and outputs are saved to ignored
`artifacts/correctness/`. Tests also cover malformed files and input validation.
Running pytest without `--v1-binary` runs the Python checks and skips C++ checks.
`--device cpu` on the runner changes only v0's device; v1 still requires a GPU
for nonempty inputs.

Individual generation and execution are also available:

```bash
PYTHONPATH=src python -m moe generate artifacts/example.bin --tokens 37 --experts 5 --k 13 --n 7 --routing hotspot
PYTHONPATH=src python -m moe reference artifacts/example.bin artifacts/example.v0.bin
build/moe_v1 artifacts/example.bin artifacts/example.v1.bin
```

Create `artifacts/` first if it does not exist. Generation accepts `--seed`
and `--counts` (E nonnegative integers summing to T) to specify expert loads.
An example for explicit counts is `--tokens 9 --experts 3 --counts 4 0 5`.

## Design documents

- [v0: PyTorch correctness reference](docs/v0/design.md)
- [v1: C++ / cuBLAS per-expert baseline](docs/v1/design.md)
- [Project proposal](docs/project_proposal/moe_proposal.pdf)

v2 grouped execution and v3 tile-based scheduling are future work.
