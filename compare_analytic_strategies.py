from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_analytic_milp import costs, scope, solve_analytic, unit_cost


METHODS = (
    ("enumerated", "Enumerated analytic B&B", "tab:blue", "o"),
    ("joint", "Joint MINLP B&B", "tab:orange", "x"),
    ("warm", "Joint B&B + relaxation warm start", "tab:green", "+"),
)


def benchmark():
    solve_analytic(.10, strategy="joint")
    records = []
    for index in range(1, 101):
        load = index / 100
        methods = METHODS[index % 3:] + METHODS[:index % 3]
        for strategy, _, _, _ in methods:
            audit = {}
            start = perf_counter()
            solution, lower, status = solve_analytic(load, strategy=strategy, audit=audit)
            record = dict(P_pu=load, strategy=strategy, seconds=perf_counter() - start,
                          lower_bound=lower, status=status, dynamic_ok=False, **audit)
            if solution is not None:
                capacity, count = solution
                slope, steady, drop = scope["frequency_metrics"](capacity, load, count)
                cost = costs @ capacity + unit_cost * count
                record.update(cost=cost, gap=cost-lower, N=count, S1_MW=capacity[0],
                              S2_MW=capacity[1], S3_MW=capacity[2], rocof_abs=abs(slope),
                              steady_abs=abs(steady), nadir_hz=50+drop,
                              dynamic_ok=abs(slope)<=.5000001 and abs(steady)<=.3000001 and drop>=-.5000001)
            records.append(record)
    return pd.DataFrame(records)


def plot_cost_time(data):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for strategy, label, color, marker in METHODS:
        rows = data[data.strategy.eq(strategy)].sort_values("P_pu")
        plotted = rows.cost.notna()
        for axis, metric in zip(axes, ("cost", "seconds")):
            axis.plot(rows.P_pu, rows[metric].where(plotted), color=color,
                      marker=marker, markersize=4, label=label)
        unresolved = plotted & rows.status.ne("gap_met")
        axes[0].scatter(rows.P_pu[unresolved], rows.cost[unresolved],
                        s=80, facecolors="none", edgecolors=color)
    axes[0].set_ylabel("Annual cost (million yuan/year)")
    axes[1].set(ylabel="Solve time (s)", yscale="log")
    return finish(fig, axes, data)


def plot_frequency(data):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    metrics = (("rocof_abs", "Initial |RoCoF| (Hz/s)", .50),
               ("steady_abs", "Steady-state |deviation| (Hz)", .30),
               ("nadir_hz", "Frequency nadir (Hz)", 49.50))
    for axis, (metric, ylabel, limit) in zip(axes, metrics):
        for strategy, label, color, marker in METHODS:
            rows = data[data.strategy.eq(strategy)].sort_values("P_pu")
            axis.plot(rows.P_pu, rows[metric], color=color, marker=marker, markersize=4, label=label)
        axis.axhline(limit, color="black", linestyle="--", label="Frequency limit")
        axis.set_ylabel(ylabel)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_ylim(.49, .51)
    return finish(fig, axes, data)


def finish(fig, axes, data):
    shown = data.loc[data.cost.notna(), "P_pu"]
    axes[-1].set(xlabel="Load step P (p.u.)", xlim=(shown.min(), shown.max()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .91))
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data = benchmark()
    data.to_csv(root / "analytic_strategy_comparison.csv", index=False)
    for plot, filename, caption in (
        (plot_cost_time, "analytic_strategy_cost_time.png", "Figure 11-1. Three analytic planning strategies: cost and time"),
        (plot_frequency, "analytic_strategy_frequency.png", "Figure 12-1. Three analytic planning strategies: frequency checks"),
    ):
        figure = plot(data)
        figure.text(.5, .01, caption, ha="center")
        figure.subplots_adjust(bottom=.14)
        figure.savefig(root / filename, dpi=160)
        plt.close(figure)
