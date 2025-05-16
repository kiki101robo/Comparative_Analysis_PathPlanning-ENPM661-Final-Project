#!/usr/bin/env python3
"""
plot_eval_cache.py
Visualises metrics stored in eval_cache.json.

Figures
-------
1  Success‑rate bar chart
2  Average runtime bar chart
3  Average jerkiness bar chart
4  Grouped bars: relative path length (per map)
5  Grouped bars: relative jerkiness (per map)
6  Scatter: runtime vs absolute path length (successful runs)
7  Line chart: runtime per map for each planner  ← new, replaces heat‑map

Run
---
python plot_eval_cache.py
python plot_eval_cache.py --save png    # save figs instead of GUI
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r") as f:
        data = json.load(f)
    if not data:
        raise ValueError("eval_cache.json is empty")
    return data


def grouped_bars(ax, values, planners, ylabel, title):
    """values: 2‑D array (maps × planners)"""
    n_maps, n_planners = values.shape
    width = 0.8 / n_planners
    offs = np.arange(n_maps)
    for j, p in enumerate(planners):
        ax.bar(
            offs + j * width,
            values[:, j],
            width,
            label=p,
        )
    ax.set_xticks(
        offs + width * (n_planners - 1) / 2,
        [f"Map {i+1}" for i in range(n_maps)],
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.margins(y=0.05)


def save_or_show(fmt):
    if fmt:
        for i, fig in enumerate(map(plt.figure, plt.get_fignums()), 1):
            fig.savefig(f"figure_{i}.{fmt}", dpi=150)
        print(f"Saved {len(plt.get_fignums())} files as *.{fmt}")
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# Main plotting routine
# --------------------------------------------------------------------------- #
def make_plots(results, out_format=None):
    planners = sorted({r["planner"] for r in results})
    n_maps = len({r["map_idx"] for r in results})
    idx = {p: i for i, p in enumerate(planners)}
    x_planners = np.arange(len(planners))

    # Arrays
    succ_rates = np.zeros(len(planners))
    avg_times = np.zeros(len(planners))
    avg_jerks = np.zeros(len(planners))

    rel_len = np.full((n_maps, len(planners)), np.inf)
    rel_jerk = np.full_like(rel_len, np.inf)
    runtime_by_map = np.full_like(rel_len, np.nan)

    # Aggregate
    for p in planners:
        runs = [r for r in results if r["planner"] == p]
        succ_rates[idx[p]] = 100 * sum(r["success"] for r in runs) / len(runs)
        avg_times[idx[p]] = np.mean([r["time"] for r in runs])
        jerks = [r["jerk"] for r in runs if r["success"]]
        avg_jerks[idx[p]] = np.mean(jerks) if jerks else np.nan

    for m in range(n_maps):
        runs = [r for r in results if r["map_idx"] == m]
        succ_runs = [r for r in runs if r["success"]]
        if succ_runs:
            best_len = min(r["length"] for r in succ_runs)
            best_jerk = min(r["jerk"] for r in succ_runs)
            for r in succ_runs:
                j = idx[r["planner"]]
                rel_len[m, j] = r["length"] / best_len
                rel_jerk[m, j] = r["jerk"] / (best_jerk or 1)
        for r in runs:
            runtime_by_map[m, idx[r["planner"]]] = r["time"]

    # -------- Fig 1 – success rate --------
    fig1, ax1 = plt.subplots()
    ax1.bar(x_planners, succ_rates)
    ax1.set_xticks(x_planners, planners, rotation=45, ha="right")
    ax1.set_ylabel("Success rate (%)")
    ax1.set_title("Planner success rate")
    fig1.tight_layout()

    # -------- Fig 2 – average runtime --------
    fig2, ax2 = plt.subplots()
    ax2.bar(x_planners, avg_times)
    ax2.set_xticks(x_planners, planners, rotation=45, ha="right")
    ax2.set_ylabel("Average runtime (s)")
    ax2.set_title("Average runtime per planner")
    fig2.tight_layout()

    # -------- Fig 3 – average jerkiness --------
    fig3, ax3 = plt.subplots()
    ax3.bar(x_planners, avg_jerks)
    ax3.set_xticks(x_planners, planners, rotation=45, ha="right")
    ax3.set_ylabel("Average jerkiness")
    ax3.set_title("Average jerkiness per planner")
    fig3.tight_layout()

    # -------- Fig 4 – relative path length --------
    fig4, ax4 = plt.subplots()
    grouped_bars(
        ax4,
        rel_len,
        planners,
        ylabel="Relative path length (1 = best)",
        title="Per‑map relative path length",
    )
    fig4.tight_layout()

    # -------- Fig 5 – relative jerkiness --------
    fig5, ax5 = plt.subplots()
    grouped_bars(
        ax5,
        rel_jerk,
        planners,
        ylabel="Relative jerkiness (1 = best)",
        title="Per‑map relative jerkiness",
    )
    fig5.tight_layout()

    # -------- Fig 6 – runtime vs absolute path length --------
    fig6, ax6 = plt.subplots()
    for p in planners:
        xs = [r["time"] for r in results if r["planner"] == p and r["success"]]
        ys = [r["length"] for r in results if r["planner"] == p and r["success"]]
        ax6.scatter(xs, ys, label=p)
    ax6.set_xlabel("Runtime (s)")
    ax6.set_ylabel("Path length (px)")
    ax6.set_title("Runtime vs path length")
    ax6.legend(fontsize=8)
    fig6.tight_layout()

    # -------- Fig 7 – runtime per map line chart --------
    # -------- Fig 7 – runtime per map (grouped bars) --------
    fig7, ax7 = plt.subplots()
    grouped_bars(
        ax7,
        runtime_by_map,  # 2‑D array: maps × planners
        planners,  # bar labels / legend
        ylabel="Runtime (s)",
        title="Runtime per map for each planner",
    )
    fig7.tight_layout()

    save_or_show(out_format)


# --------------------------------------------------------------------------- #
# CLI wrapper
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Plot evaluation cache metrics.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).with_name("eval_cache.json"),
        help="Path to eval_cache.json",
    )
    parser.add_argument(
        "--save",
        metavar="FMT",
        help="Save figures instead of showing (e.g. png, pdf, svg)",
    )
    args = parser.parse_args()

    make_plots(load_results(args.cache), out_format=args.save)


if __name__ == "__main__":
    main()
