#!/usr/bin/env python3
"""Measure the v0 packed expert loop on one CUDA stream."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe.reference import reference_fp64, run_experts
from moe.workloads import generate

SEED = 618
WARMUP = 20
SAMPLES = 30
REPEATS = 20
TOKENS = (32, 256, 1024, 4096)
EXPERTS = (8, 32, 64)
SHAPES = ((256, 256), (512, 1024))
ROUTINGS = ("balanced", "uniform", "hotspot")
FIXED_ACTIVE_COUNTS = {"balanced": (32,) * 8, "skewed": (144,) + (16,) * 7}


def measure(t, e, k, n, routing, stream, *, check_fp64, counts=None):
    """Prepare one workload and return its per-round timing samples and summary."""
    workload = generate(t, e, k, n, routing=routing, seed=SEED, counts=counts)
    with torch.cuda.stream(stream):
        x = torch.as_tensor(workload.x, device="cuda")
        w = torch.as_tensor(workload.w, device="cuda")
        y = torch.empty((t, n), dtype=torch.float32, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # CUDA events initialize lazily. Record both before any measured batch.
        start.record(stream)
        for _ in range(WARMUP):
            run_experts(x, w, y, workload.offsets)
        end.record(stream)
        end.synchronize()
        if check_fp64:
            np.testing.assert_allclose(
                y.cpu().numpy(), reference_fp64(workload), atol=1e-4, rtol=1e-4
            )

        times = []
        for _ in range(SAMPLES):
            host_start = perf_counter_ns()
            start.record(stream)
            for _ in range(REPEATS):
                run_experts(x, w, y, workload.offsets)
            end.record(stream)
            end.synchronize()
            host_us = (perf_counter_ns() - host_start) / (1000 * REPEATS)
            cuda_us = start.elapsed_time(end) * 1000 / REPEATS
            times.append((cuda_us, host_us))
        if not torch.isfinite(y).all().item():
            raise RuntimeError("v0 produced non-finite output")

    values = np.asarray(times)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError(f"invalid timings for {(t, e, k, n, routing)}")
    median_cuda, median_host = np.median(values, axis=0)
    p90_cuda, p90_host = np.percentile(values, 90, axis=0, method="linear")

    config = dict(t=t, e=e, k=k, n=n, routing=routing, seed=SEED)
    samples = [
        dict(config, sample=i, cuda_us=cuda_us, host_us=host_us)
        for i, (cuda_us, host_us) in enumerate(times, start=1)
    ]
    counts = np.diff(workload.offsets)
    summary = dict(
        config,
        active_experts=int(np.count_nonzero(counts)),
        expert_tokens=json.dumps(counts.tolist(), separators=(",", ":")),
        cuda_median_us=median_cuda,
        cuda_p90_us=p90_cuda,
        host_median_us=median_host,
        host_p90_us=p90_host,
        effective_gflops=2 * t * k * n / (median_cuda * 1000),
    )
    return samples, summary


def command_output(*args):
    """Capture a version or device query for the run metadata."""
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def metadata(smoke):
    """Record experiment settings and the environment that produced the results."""
    status = command_output("git", "status", "--porcelain")
    return dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=command_output("git", "rev-parse", "HEAD"),
        git_dirty=bool(status),
        git_status=status.splitlines(),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/moe/reference.py",
                ROOT / "src/moe/workloads.py",
                ROOT / "src/moe/format.py",
            )
        },
        command=[sys.executable, *sys.argv],
        mode="smoke" if smoke else "full",
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        pytorch=str(torch.__version__),
        pytorch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        gpu_compute_capability=list(torch.cuda.get_device_capability()),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        nvidia_smi=command_output(
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ).splitlines(),
        dtype="float32",
        tf32=False,
        cuda_streams=1,
        seed=SEED,
        warmup_rounds=WARMUP,
        samples_per_config=SAMPLES,
        rounds_per_sample=REPEATS,
        tokens=(TOKENS[0],) if smoke else TOKENS,
        experts=(EXPERTS[0],) if smoke else EXPERTS,
        shapes=(SHAPES[0],) if smoke else SHAPES,
        routings=(ROUTINGS[0],) if smoke else ROUTINGS,
    )


def write_csv(path, rows):
    """Write rows using the first row's keys as the header."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the CUDA-only smoke test or full v0 matrix and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Measure one case and validate against CPU FP64")
    mode.add_argument("--fixed-active", action="store_true", help="Compare two loads with exactly eight active experts")
    parser.add_argument("--output-dir", type=Path, help="Directory for the CSV and metadata files")
    parser.add_argument("--order-seed", type=int, help="Shuffle configuration order with this seed")
    args = parser.parse_args()
    if args.fixed_active and args.output_dir is None:
        parser.error("--fixed-active requires --output-dir to store each independent run")
    if not torch.cuda.is_available():
        parser.error("CUDA is required to benchmark v0")
    output = args.output_dir
    if output is None:
        output = ROOT / ("artifacts/benchmark_v0_smoke" if args.smoke else "results/v0")

    info = metadata(args.smoke)
    if args.fixed_active:
        # Explicit counts skip routing RNG draws, so both cases use identical X and W.
        info.update(mode="fixed-active", tokens=[256], experts=[8], shapes=[(256, 256)],
                    routings=list(FIXED_ACTIVE_COUNTS), explicit_counts=FIXED_ACTIVE_COUNTS)
    configs = list(product(info["tokens"], info["experts"], info["shapes"], info["routings"]))
    info["order_seed"] = args.order_seed
    if args.order_seed is not None:
        random.Random(args.order_seed).shuffle(configs)
    all_samples, summaries = [], []
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        stream = torch.cuda.Stream()
        with torch.inference_mode():
            for i, (t, e, (k, n), routing) in enumerate(configs, start=1):
                counts = FIXED_ACTIVE_COUNTS[routing] if args.fixed_active else None
                samples, summary = measure(t, e, k, n, routing, stream,
                                           check_fp64=args.smoke or args.fixed_active, counts=counts)
                all_samples.extend(samples)
                summaries.append(summary)
                if args.smoke or i % 12 == 0:
                    print(f"Measured {i}/{len(configs)} configurations", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "samples.csv", all_samples)
    write_csv(output / "summary.csv", summaries)
    (output / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(summaries)} configurations and {len(all_samples)} samples to {output}")


if __name__ == "__main__":
    main()
