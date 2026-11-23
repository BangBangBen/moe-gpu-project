#!/usr/bin/env python3
"""Plot v0 benchmark retests from their per-run summary CSVs."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"balanced": "#2563a6", "uniform": "#278269", "hotspot": "#c45228"}
METRICS = ("host_median_us", "cuda_median_us")


def load_runs(retests):
    """Load matching configurations and timing medians from independent runs."""
    runs = []
    for path in sorted(retests.glob("seed-*/summary.csv")):
        rows = {}
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                key = (*(int(row[name]) for name in ("t", "e", "k", "n")), row["routing"], int(row["seed"]))
                timings = tuple(float(row[name]) for name in METRICS)
                active = int(row["active_experts"])
                if key in rows or not np.isfinite(timings).all() or min(timings) <= 0:
                    raise ValueError(f"Duplicate configuration or invalid timing in {path}: {key}")
                rows[key] = (*timings, active)
        runs.append(rows)
    if len(runs) < 2 or not runs[0]:
        raise ValueError(f"Expected at least two nonempty retest summaries under {retests}")
    if any(run.keys() != runs[0].keys() for run in runs[1:]):
        raise ValueError("Retest configurations do not match")
    if any(run[key][2] != runs[0][key][2] for run in runs[1:] for key in run):
        raise ValueError("Active expert counts do not match across retests")
    if {key[5] for key in runs[0]} != {618}:
        raise ValueError("Expected workload seed 618")
    return {key[:5]: np.array([run[key] for run in runs]) for key in runs[0]}


def save(fig, output, name):
    """Save a figure as a standalone SVG."""
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def latency_by_tokens(data, output, tokens, experts, shapes):
    """Show the median of run medians and the min–max range across runs."""
    fig, axes = plt.subplots(len(shapes), len(experts), figsize=(11, 6), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for ax, ((k, n), e) in zip(axes.flat, product(shapes, experts)):
        for routing, color in COLORS.items():
            times = np.array([data[t, e, k, n, routing][:, 0] for t in tokens]) / 1000
            ax.fill_between(tokens, times.min(axis=1), times.max(axis=1), color=color, alpha=.14)
            ax.plot(tokens, np.median(times, axis=1), "o-", color=color, label=routing)
        ax.set(title=f"E={e}  ·  K={k}, N={n}", xscale="log", xticks=tokens, xticklabels=tokens)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Tokens (log scale)")
    fig.supylabel("Host latency per round (ms)")
    fig.suptitle("v0 latency by token count\nLines show the median across runs. Shading shows the run range.")
    save(fig, output, "latency_by_tokens")


def latency_by_active_experts(data, output, tokens, experts, shapes):
    """Compare host latency with the number of active experts at fixed T and shape."""
    fig, axes = plt.subplots(len(shapes), len(tokens), figsize=(12, 6), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for ax, ((k, n), t) in zip(axes.flat, product(shapes, tokens)):
        for routing, color in COLORS.items():
            cases = [data[t, e, k, n, routing] for e in experts]
            ax.scatter([case[0, 2] for case in cases], [np.median(case[:, 0]) / 1000 for case in cases],
                       marker={"balanced": "o", "uniform": "^", "hotspot": "x"}[routing],
                       color=color, label=routing)
        ax.set_title(f"T={t}  ·  K={k}, N={n}", fontsize=10)
        ax.grid(alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Active experts")
    fig.supylabel("Median host latency per round (ms)")
    fig.suptitle("v0 latency by active expert count")
    save(fig, output, "latency_by_active_experts")


def hotspot_heatmap(data, output, tokens, experts, shapes):
    """Show the median of paired, within-run hotspot-to-balanced latency ratios."""
    matrices = [
        np.array([[np.median(data[t, e, k, n, "hotspot"][:, 0] /
                             data[t, e, k, n, "balanced"][:, 0])
                   for t in tokens] for e in experts])
        for k, n in shapes
    ]
    extent = max(.01, max(np.max(np.abs(m - 1)) for m in matrices))
    norm = TwoSlopeNorm(vmin=1 - extent, vcenter=1, vmax=1 + extent)
    fig, axes = plt.subplots(1, len(shapes), figsize=(9, 3.5), squeeze=False, layout="constrained")
    for ax, (k, n), ratios in zip(axes.flat, shapes, matrices):
        im = ax.imshow(ratios, norm=norm, cmap="RdBu_r", aspect="auto")
        for (row, col), ratio in np.ndenumerate(ratios):
            ink = "white" if abs(ratio - 1) > .55 * extent else "#17202a"
            ax.text(col, row, f"{ratio:.2f}×", ha="center", va="center", color=ink)
        ax.set(title=f"K={k}, N={n}", xticks=range(len(tokens)), xticklabels=tokens,
               yticks=range(len(experts)), yticklabels=experts, xlabel="Tokens", ylabel="Experts")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Host latency ratio")
    fig.suptitle("v0 hotspot / balanced host latency")
    save(fig, output, "hotspot_vs_balanced")


def timer_agreement(data, output):
    """Compare CUDA Events and host medians for every configuration and run."""
    cases = np.stack(list(data.values()))[:, :, :2] / 1000
    fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
    limits = [cases.min() * .92, cases.max() * 1.08]
    ax.plot(limits, limits, "--", color="#555555", label="Equal timing")
    for run in range(cases.shape[1]):
        color = plt.get_cmap("tab10")(run % 10)
        ax.scatter(*cases[:, run, :].T, s=17, alpha=.45, color=color, label=f"Retest {run + 1}")
    ax.set(xlabel="Host median per round (ms)", ylabel="CUDA Events median per round (ms)",
           title="v0 timer agreement", xlim=limits, ylim=limits, aspect="equal")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    save(fig, output, "timer_agreement")


def fixed_active_plot(runs, output):
    """Plot expert loads and paired host medians for the fixed-active control."""
    data = load_runs(runs)
    keys = [(256, 8, 256, 256, mode) for mode in ("balanced", "skewed")]
    if set(data) != set(keys) or any((data[key][:, 2] != 8).any() for key in keys):
        raise ValueError("Expected the balanced and skewed cases with eight active experts")
    summaries = sorted(runs.glob("seed-*/summary.csv"))
    with summaries[0].open(newline="") as file:
        counts = {row["routing"]: json.loads(row["expert_tokens"])
                  for row in csv.DictReader(file)}
    fig, (load_ax, time_ax) = plt.subplots(1, 2, figsize=(9.4, 4.2), layout="constrained")
    positions = np.arange(8)
    for i, mode in enumerate(("balanced", "skewed")):
        color = COLORS["balanced" if i == 0 else "hotspot"]
        load_ax.bar(positions + (i - .5) * .36, counts[mode], width=.36, color=color, label=mode)
    load_ax.set(xlabel="Expert ID", ylabel="Tokens assigned", xticks=positions, title="Expert workloads")
    load_ax.legend(frameon=False)
    times = np.stack([data[key][:, 0] for key in keys])
    for i, run in enumerate(times.T):
        time_ax.plot([0, 1], run, "o-", color="#9aa3ad", alpha=.65,
                     label="One independent run" if i == 0 else None)
    time_ax.plot([0, 1], np.median(times, axis=1), "o-", color="#24354b", lw=2,
                 label="Median of run medians")
    time_ax.set(xticks=[0, 1], xticklabels=["Balanced", "Skewed"], xlim=(-.25, 1.25),
                ylabel="Host median per round (µs)", title="Paired latency measurements")
    time_ax.legend(frameon=False)
    for ax in (load_ax, time_ax):
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("v0 fixed-active control · T=256, E=8, K=N=256")
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "fixed_active_comparison")


def main():
    """Generate the grid figures or the fixed-active comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retests", type=Path, help="Directory containing independent seed-* runs")
    parser.add_argument("--fixed-active", action="store_true", help="Plot the fixed-active experiment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/v0/plots")
    args = parser.parse_args()
    if args.fixed_active:
        fixed_active_plot(args.retests or ROOT / "results/v0/fixed_active", args.output_dir)
        return
    data = load_runs(args.retests or ROOT / "results/v0/retests")
    tokens = sorted({t for t, _, _, _, _ in data})
    experts = sorted({e for _, e, _, _, _ in data})
    shapes = sorted({(k, n) for _, _, k, n, _ in data})
    expected = {(t, e, k, n, r) for t, e, (k, n), r in product(tokens, experts, shapes, COLORS)}
    if data.keys() != expected:
        raise ValueError("Retests must contain the complete configuration grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency_by_tokens(data, args.output_dir, tokens, experts, shapes)
    latency_by_active_experts(data, args.output_dir, tokens, experts, shapes)
    hotspot_heatmap(data, args.output_dir, tokens, experts, shapes)
    timer_agreement(data, args.output_dir)


if __name__ == "__main__":
    main()
