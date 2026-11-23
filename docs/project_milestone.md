# Project Milestone

**Progressive Optimization of Mixture-of-Experts Inference on GPUs**  
**CMU 15-418/15-618 Final Project**  
**Team:** Bruce Cheng and Ben Chai  
**Date:** November 23, 2026

## Current Progress

We set up the development environment, benchmark harness, and experiment
automation, then implemented and validated two FP32 MoE expert GEMM baselines.
V0 is a PyTorch correctness reference that slices packed tokens by expert
and calls `torch.mm` once per active expert. V1 is a C++ baseline that
submits one cuBLAS SGEMM per active expert on a single CUDA stream.
We completed the V0 benchmarks on a local NVIDIA GeForce RTX 2080 SUPER
and documented the setup, results, and reproduction steps. V1 benchmarking
is the next step. [V0 experiment report](v0/report.md)

With the total fixed at 256 tokens, increasing the active expert count
from 4 to 64 raised V0's synchronized host latency from 102 µs to 2,559 µs.
CUDA Graph replay substantially reduced latency, though replay latency
still rose as the work was split across more experts. With eight active
experts, light and moderate skew had little effect. High skew raised the
median paired host latency by 48.7% and doubled the observed GPU kernel
count from 8 to 16 with the same eight `torch.mm` calls. These results
motivate reducing repeated host submission and testing how grouped
execution handles uneven GEMM shapes.

## Updated Schedule

Completed work includes preliminary exploration, toolchain and environment
setup, benchmark automation, V0 implementation and measurements, and
V1 implementation and correctness validation.

| Week of | Planned work |
| --- | --- |
| November 23 | Benchmark V1 (Ben). Move the test harness to the GHC machines with NVIDIA RTX 2080 GPUs and check whether the local trends reproduce. Implement V2 grouped execution in C++ using cuBLAS (Bruce). |
| November 30 | Benchmark V2 (Bruce). Analyze the measurements and implement V3 tile-based execution with a custom CUTLASS scheduler (Ben). |
| December 7 | Benchmark V3 (Ben). Analyze and organize the measurements, complete the final report, and prepare the poster session. |

## Poster Session Deliverables

- Background on MoE language models and the performance challenges that
  motivate the project.
- Algorithm design for each version and the shared experimental setup.
- Benchmark and profiling results, with performance comparisons across
  versions.
- Analysis of how the measurements informed the move from per-expert
  execution to grouping and then to tile scheduling.

## Assessment of Current Progress

The initial environment and benchmark setup took longer than expected, so
we are slightly behind the original schedule. With both baselines
validated and the V0 experiments complete, the project remains broadly
on track.

## Anticipated Challenges

- **Reproducibility on GHC.** We plan to use GHC as the reference environment
  for cross-version comparisons if measurements are consistent. We still
  need to verify that the trends reproduce there. The local setup uses WSL2
  on Windows 11 Home with an RTX 2080 SUPER, while the GHC machines use an
  RTX 2080. We will report measurements by platform.
- **Custom CUTLASS scheduling.** V3 requires a scheduler that maps tiles
  from differently sized expert GEMMs to GPU thread blocks. We will need
  to check correctness and measure whether the scheduling overhead pays
  off under uneven expert loads.
