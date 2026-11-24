# Progressive Optimization of Mixture-of-Experts Inference on GPUs

**CMU 15-418/15-618 Final Project**

**Team:** Bruce Cheng and Ben Chai

## Project Overview

We investigate GPU work scheduling for the expert computation in
Mixture-of-Experts inference. Starting from a per-expert baseline, we
will progressively explore grouped execution and tile-based scheduling,
and evaluate how workload imbalance affects performance.

## Project Documents

- [Project Proposal](./docs/project_proposal/moe_proposal.pdf)
- [v0 Experiment Report](./docs/v0/report.md)
- [Milestone Report](./docs/project_milestone.md)
- Final Report — coming soon
- Poster — coming soon

## Schedule and Progress

Completed: environment and benchmark setup, V0 implementation and measurements, and V1 implementation. See the [milestone report](./docs/project_milestone.md) for details.

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
