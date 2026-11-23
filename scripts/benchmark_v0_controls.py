#!/usr/bin/env python3
"""Run one independent v0 control experiment or trace its GPU work."""

import argparse
from functools import partial
import hashlib
from itertools import accumulate
import json
from pathlib import Path
import random
from time import perf_counter_ns

import numpy as np
import torch

from benchmark_v0 import ROOT, REPEATS, SAMPLES, SEED, WARMUP, metadata, write_csv
from moe.reference import reference_fp64, run_experts
from moe.workloads import generate

TOKENS, EXPERTS, WIDTH = 256, 64, 256
ACTIVE_COUNTS = {
    str(n): (TOKENS // n,) * n + (0,) * (EXPERTS - n) for n in (4, 8, 16, 32, 64)
}
SKEW_COUNTS = {
    label: (largest,) + (rest,) * 7 + (0,) * (EXPERTS - 8)
    for label, largest, rest in (("balanced", 32, 32), ("light", 67, 27),
                                 ("medium", 102, 22), ("high", 144, 16))
}


def offsets_for(counts):
    """Convert the fixed experiment counts into expert boundaries."""
    return np.array(list(accumulate(counts, initial=0)), dtype=np.int64)


def batch_times(invoke, stream):
    """Measure repeated expert rounds with the original v0 timing boundaries."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    end.record(stream)
    end.synchronize()
    for _ in range(SAMPLES):
        host_start = perf_counter_ns()
        start.record(stream)
        for _ in range(REPEATS):
            invoke()
        end.record(stream)
        end.synchronize()
        host = (perf_counter_ns() - host_start) / (1000 * REPEATS)
        yield host, start.elapsed_time(end) * 1000 / REPEATS


def submit_times(invoke, stream):
    """Time submission of one expert round with no queued work at its start."""
    stream.synchronize()
    for _ in range(SAMPLES):
        start = perf_counter_ns()
        invoke()
        host = (perf_counter_ns() - start) / 1000
        stream.synchronize()
        yield host


def validate_output(y, expected):
    """Check a completed GPU output against the independent CPU FP64 result."""
    np.testing.assert_allclose(y.cpu().numpy(), expected, atol=1e-4, rtol=1e-4)


def measure_case(workload, buffers, stream, args, case, counts, case_position):
    """Validate, warm, and measure each execution path for one routed workload."""
    x, w, y = buffers
    workload.offsets = offsets = offsets_for(counts)
    expected = reference_fp64(workload)
    eager = partial(run_experts, x, w, y, offsets)
    paths = {"eager": eager}
    if args.experiment == "active":
        # Warm the eager GEMMs on the capture stream before making the graph.
        for _ in range(WARMUP):
            eager()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            eager()
        paths["graph"] = graph.replay

    samples, submissions, summaries = [], [], []
    order = list(paths)
    if args.reverse:
        order.reverse()
    for path_position, name in enumerate(order, 1):
        invoke = paths[name]
        y.fill_(float("nan"))
        for _ in range(WARMUP):
            invoke()
        stream.synchronize()
        validate_output(y, expected)
        if args.trace:
            with torch.cuda.nvtx.range(f"v0-control/{args.experiment}/{case}/{name}"):
                invoke()
                stream.synchronize()
            continue

        config = dict(experiment=args.experiment, case=case, path=name, order_seed=args.order_seed,
                      reverse=int(args.reverse), case_position=case_position, path_position=path_position,
                      active_experts=sum(c > 0 for c in counts), max_token_share=max(counts) / TOKENS)
        times = np.array(list(batch_times(invoke, stream)))
        if not np.isfinite(times).all() or (times <= 0).any():
            raise RuntimeError(f"Invalid batch timings for {config}")
        samples.extend(dict(config, sample=i, host_us=host, cuda_us=cuda)
                       for i, (host, cuda) in enumerate(times, 1))
        summary = dict(config, expert_tokens=json.dumps(counts, separators=(",", ":")))
        for metric, values in zip(("host", "cuda"), times.T):
            summary[f"{metric}_median_us"] = np.median(values)
            summary[f"{metric}_p90_us"] = np.percentile(values, 90)
        if args.experiment == "active":
            submit = np.array(list(submit_times(invoke, stream)))
            if not np.isfinite(submit).all() or (submit <= 0).any():
                raise RuntimeError(f"Invalid submission timings for {config}")
            submissions.extend(dict(config, sample=i, submit_us=us) for i, us in enumerate(submit, 1))
            summary.update(submit_median_us=np.median(submit), submit_p90_us=np.percentile(submit, 90))
        summaries.append(summary)
    return samples, submissions, summaries


def main():
    """Run one process with seeded case order, optionally reversed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("active", "skew"), required=True)
    parser.add_argument("--order-seed", type=int, default=101)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trace", action="store_true", help="Mark one GPU round per case and path for Nsight")
    args = parser.parse_args()
    if not args.trace and args.output_dir is None:
        parser.error("--output-dir is required for timing")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    cases = ACTIVE_COUNTS if args.experiment == "active" else SKEW_COUNTS
    order = list(cases)
    if not args.trace:
        random.Random(args.order_seed).shuffle(order)
    if args.reverse:
        order.reverse()

    if not args.trace:
        info = metadata(False)
        for field in ("tokens", "experts", "shapes", "routings"):
            del info[field]
        source = Path(__file__).resolve()
        info["source_sha256"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        info.update(mode="control", experiment=args.experiment, tokens=TOKENS, experts=EXPERTS,
                    k=WIDTH, n=WIDTH, order_seed=args.order_seed, reverse=args.reverse,
                    case_order=order, counts=cases, submission_samples=SAMPLES if args.experiment == "active" else 0)

    workload = generate(TOKENS, EXPERTS, WIDTH, WIDTH, seed=SEED, counts=next(iter(cases.values())))
    previous = torch.backends.cuda.matmul.allow_tf32
    stream = torch.cuda.Stream()
    samples, submissions, summaries = [], [], []
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.cuda.stream(stream):
            buffers = (torch.as_tensor(workload.x, device="cuda"), torch.as_tensor(workload.w, device="cuda"),
                       torch.empty((TOKENS, WIDTH), device="cuda", dtype=torch.float32))
            for i, case in enumerate(order, 1):
                batch, submitted, summary = measure_case(workload, buffers, stream, args, case, cases[case], i)
                samples.extend(batch)
                submissions.extend(submitted)
                summaries.extend(summary)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    if not args.trace:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "samples.csv", samples)
        write_csv(args.output_dir / "summary.csv", summaries)
        if submissions:
            write_csv(args.output_dir / "submission.csv", submissions)
        (args.output_dir / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"{args.output_dir}: {len(samples)} batch samples, {len(submissions)} submission samples")


if __name__ == "__main__":
    main()
