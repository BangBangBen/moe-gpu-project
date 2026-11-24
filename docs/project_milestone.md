# Progressive Optimization of Mixture-of-Experts Inference on GPUs
**15-618 Project Milestone**
Team: Bruce Cheng and Ben Chai
Project URL: https://bangbangben.github.io/moe-gpu-project/

## Current Progress
We implemented and validated two FP32 MoE expert GEMM baselines. V0 is a PyTorch reference that slices packed tokens by expert and calls torch.mm once per active expert. V1 is a C++ baseline that submits one cuBLAS SGEMM per active expert on a single CUDA stream. We completed and documented the v0 performance experiments, and v1 performance benchmarking is next.
With 256 tokens fixed, increasing the active expert count from 4 to 64 raised v0’s synchronized host latency from 102 µs to 2,559 µs. CUDA Graph replay substantially reduced latency, though replay latency still rose as the work was split across more experts. With eight active experts, light and moderate skew had little effect, while high skew increased median paired latency by 48.7% and doubled the observed GPU kernel count from 8 to 16. These findings motivate reducing repeated host submission and testing how grouped execution handles uneven GEMM shapes.

## Updated Schedule
Completed:
- Preliminary exploration, toolchain and environment setup
- Benchmark and automation setup
- V0: PyTorch naive implementation as correctness baseline
- V0 benchmark and measurements (on local NVIDIA RTX 2080 Super)
- V1: cuBLAS C++ naive implementation
Up Next:
- Nov. 23–25
   - V1 benchmark and measurements; record the baseline results. (Ben)
   - Move the benchmark to GHC machines and check reproducibility. (Bruce)
- Nov. 26–29
   - Implement and validate V2: grouped execution using C++ / cuBLAS. (Bruce)
- Nov. 30–Dec. 2
   - Benchmark and profile V2. (Bruce)
   - Begin V3: tile-based scheduling using CUTLASS with a custom scheduler. (Ben)
- Dec. 3–6
   - Continue V3 implementation and correctness validation. (Ben)
- Dec. 7–9
   - Benchmark and profile V3. (Ben)
   - Analyze and organize measurements; prepare the final report and poster. (Bruce)
- Dec. 10–11
   - Submit the final report by Dec. 10, 11:59pm; prepare for the Dec. 11 poster session. (together)

## Deliverables for Poster Session
- Project context and motivation: background on MoE LLMs and current landscape on performance optimization challenges for these types of LLMs.
- Algorithm designs for each version
- Experiment setup
- Benchmark and profiling results
- Performance comparison for each version
- Analysis on how profiling/benchmarking results inspired the optimization technique used in the algorithm design

## Assessment on Current Progress
- We are a little behind schedule due to the complexity of the initial environment and benchmark setup. Grouped execution, originally scheduled for Nov. 16–22, is now planned for Nov. 26–29. The revised schedule retains grouped execution and tile-based scheduling as planned goals. We are largely on track.

## Anticipated Challenges
- Consistently reproducing benchmark results on GHC machines is unknown right now. But ideally we should use GHC machines as a reference since it has a cleaner environment than the local machine (WSL 2 on Windows 11 Home).
- Tile-based custom scheduling via CUTLASS might present some challenges in algorithm design.
