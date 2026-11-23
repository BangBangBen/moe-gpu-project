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

- **Nov. 9–15:** Set up the benchmark and reference implementation;
  implement and validate the per-expert baseline.
- **Nov. 16–22:** Profile the baseline and develop grouped execution.
- **Nov. 23–29:** Benchmark V1, reproduce measurements on GHC, and implement V2 grouped execution.
- **Nov. 30–Dec. 6:** Benchmark V2 and implement V3 tile-based scheduling.
- **Dec. 7–11:** Benchmark V3, finalize the report, and prepare the poster.
