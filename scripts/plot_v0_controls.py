#!/usr/bin/env python3
"""Validate, summarize, and plot the v0 active-expert and skew controls."""

import argparse
import csv
import json
from pathlib import Path
import random
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_v0_controls import ACTIVE_COUNTS, SKEW_COUNTS, ROOT, SAMPLES
from plot_v0 import save

ORDER_SEEDS = (101, 202, 303, 404)


def read_csv(path):
    """Read a benchmark table as a list of rows."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_samples(rows, table, fields):
    """Check sample indices, positive timings, and both reported statistics."""
    keys = {(row["case"], row["path"]) for row in rows}
    if {(row["case"], row["path"]) for row in table} != keys:
        raise ValueError("Unexpected sample configuration")
    for row in rows:
        group = [s for s in table if (s["case"], s["path"]) == (row["case"], row["path"])]
        if [int(s["sample"]) for s in group] != list(range(1, SAMPLES + 1)):
            raise ValueError("Missing or invalid sample indices")
        for field in fields:
            times = np.array([float(s[f"{field}_us"]) for s in group])
            stats = [float(row[f"{field}_{stat}_us"]) for stat in ("median", "p90")]
            if (times <= 0).any() or not np.isfinite(times).all() or not np.allclose(
                stats, [np.median(times), np.percentile(times, 90)]
            ):
                raise ValueError("Invalid timings or summary statistics")


def load_runs(root, experiment, cases):
    """Read exactly eight planned runs and verify orders and controlled counts."""
    data = {}
    paths = ("eager", "graph") if experiment == "active" else ("eager",)
    for seed in ORDER_SEEDS:
        order = list(cases)
        random.Random(seed).shuffle(order)
        for reverse in (False, True):
            label = f"seed-{seed}-{'reverse' if reverse else 'forward'}"
            folder = root / experiment / label
            info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if (info["order_seed"], info["reverse"], info["tokens"], info["experts"], info["k"], info["n"]) != (
                seed, reverse, 256, 64, 256, 256
            ):
                raise ValueError(f"Unexpected experiment metadata in {folder}")
            rows = read_csv(folder / "summary.csv")
            actual = [(row["case"], row["path"]) for row in rows]
            expected = [(case, path) for case in (order[::-1] if reverse else order)
                        for path in (paths[::-1] if reverse else paths)]
            if actual != expected:
                raise ValueError(f"Unexpected execution order in {folder}")
            for row in rows:
                counts = tuple(json.loads(row["expert_tokens"]))
                if counts != cases[row["case"]] or int(row["active_experts"]) != sum(c > 0 for c in counts):
                    raise ValueError(f"Unexpected expert counts in {folder}")
            validate_samples(rows, read_csv(folder / "samples.csv"), ("host", "cuda"))
            if experiment == "active":
                validate_samples(rows, read_csv(folder / "submission.csv"), ("submit",))
            data[label] = {(row["case"], row["path"]): row for row in rows}
    return data


def timing_matrix(data, cases, path, metric):
    """Return a matrix with one run per row and one configuration per column."""
    return np.array([[float(run[case, path][f"{metric}_median_us"]) for case in cases]
                     for run in data.values()])


def spread(values):
    """Summarize the distribution of measurements across independent runs."""
    return dict(median=float(np.median(values)), min=float(np.min(values)), max=float(np.max(values)))


def active_figure(active, out):
    """Plot synchronized host latency and isolated submit latency."""
    counts = list(map(int, ACTIVE_COUNTS))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    stats = {}
    for ax, metric, title in zip(axes, ("host", "submit"), ("Synchronized host latency", "Single-round submission")):
        stats[metric] = {}
        for path, color in (("eager", "#2563a6"), ("graph", "#c45228")):
            times = timing_matrix(active, ACTIVE_COUNTS, path, metric)
            ax.scatter(np.tile(counts, len(times)), times.ravel(), s=17, color=color, alpha=.30)
            ax.plot(counts, np.median(times, axis=0), "o-", color=color, label=path)
            stats[metric][path] = {str(n): spread(values) for n, values in zip(counts, times.T)}
        ax.set(xscale="log", xticks=counts, xticklabels=counts, xlabel="Active experts", ylabel="Time per round (µs)",
               title=title)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("v0 · Fixed token count and balanced active experts")
    save(fig, out, "active_expert_control")
    for metric in ("host", "submit"):
        eager = timing_matrix(active, ACTIVE_COUNTS, "eager", metric)
        graph = timing_matrix(active, ACTIVE_COUNTS, "graph", metric)
        stats[metric]["paired_eager_over_graph"] = {str(n): spread(v) for n, v in zip(counts, (eager / graph).T)}
    return stats


def skew_figure(skew, out):
    """Plot host medians normalized to the balanced case in the same run."""
    shares = [100 * max(counts) / sum(counts) for counts in SKEW_COUNTS.values()]
    host = timing_matrix(skew, SKEW_COUNTS, "eager", "host")
    ratios = host / host[:, :1]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    for i, run in enumerate(ratios):
        ax.plot(shares, run, "o-", color="#919ca8", alpha=.5, label="Independent run" if i == 0 else None)
    ax.plot(shares, np.median(ratios, axis=0), "o-", color="#2563a6", lw=2.2, label="Median of paired ratios")
    ax.axhline(1, color="#636363", linestyle="--", linewidth=1)
    ax.set(xticks=shares, xticklabels=[f"{x:.1f}%" for x in shares],
           xlabel="Token share of busiest expert", ylabel="Host latency / same-run balanced",
           title="v0 · Eight active experts with increasing skew")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    save(fig, out, "skew_control")
    return {case: dict(host_us=spread(times), paired_over_balanced=spread(ratio),
                       runs_slower_than_balanced=int((ratio > 1).sum()))
            for case, times, ratio in zip(SKEW_COUNTS, host.T, ratios.T)}


def trace_summary(trace_dir, output):
    """Export the GPU kernel names, launch shapes, and counts from marked rounds."""
    kernels = {}
    for experiment, cases in (("active", ACTIVE_COUNTS), ("skew", SKEW_COUNTS)):
        path = trace_dir / f"{experiment}.sqlite"
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for case in cases:
                for variant in (("eager", "graph") if experiment == "active" else ("eager",)):
                    label = f"v0-control/{experiment}/{case}/{variant}"
                    ranges = conn.execute(
                        "SELECT start, end, globalTid FROM NVTX_EVENTS WHERE text = ?", (label,)
                    ).fetchall()
                    if len(ranges) != 1:
                        raise ValueError(f"Expected exactly one trace range for {label}")
                    # Correlate launches on the host clock so GPU clock offsets
                    # cannot cause a kernel to fall outside its enclosing range.
                    rows = conn.execute("""
                        SELECT name.value, gridX, gridY, gridZ, blockX, blockY, blockZ, count(*)
                        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS name ON name.id = k.demangledName
                        WHERE k.correlationId IN (
                            SELECT r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
                            WHERE r.start >= ? AND r.start <= ? AND r.globalTid = ?
                        )
                        GROUP BY name.value, gridX, gridY, gridZ, blockX, blockY, blockZ
                        ORDER BY name.value, gridX, gridY, gridZ, blockX, blockY, blockZ
                    """, ranges[0]).fetchall()
                    if not rows:
                        raise ValueError(f"No GPU kernels found for {label}")
                    kernels[label] = {"count": sum(row[-1] for row in rows),
                                      "launches": [dict(kernel=row[0], grid=row[1:4], block=row[4:7], count=row[7])
                                                   for row in rows]}
    (output / "trace_summary.json").write_text(json.dumps(kernels, indent=2) + "\n", encoding="utf-8")


def main():
    """Validate all timed runs, then write two figures and aggregate statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/v0/controls")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/v0/plots")
    parser.add_argument("--trace-dir", type=Path, help="Optional folder containing active.sqlite and skew.sqlite")
    args = parser.parse_args()
    active = load_runs(args.results, "active", ACTIVE_COUNTS)
    skew = load_runs(args.results, "skew", SKEW_COUNTS)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = dict(active=active_figure(active, out), skew=skew_figure(skew, out))
    (args.results / "analysis.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.trace_dir:
        trace_summary(args.trace_dir, args.results)


if __name__ == "__main__":
    main()
