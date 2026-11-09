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
- Milestone Report — coming soon
- Final Report — coming soon
- Poster — coming soon

## Schedule and Progress

- **Nov. 9–15:** Set up the benchmark and reference implementation;
  implement and validate the per-expert baseline.
- **Nov. 16–22:** Profile the baseline and develop grouped execution.
- **Nov. 23–29:** Submit the milestone report and work on tile-based scheduling.
- **Nov. 30–Dec. 6:** Run experiments and analyze performance bottlenecks.
- **Dec. 7–11:** Finalize the report and prepare the poster.