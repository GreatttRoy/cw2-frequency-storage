import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from heapq import heappop, heappush
from scipy.optimize import linprog, minimize


root = Path(__file__).resolve().parent
sources = ["".join(cell["source"]) for cell in json.loads(
    (root / "cw2_Q3加入储能和规划.ipynb").read_text()
)["cells"]]
scope = {"np": np, "pd": pd, "display": print}


def load_cell(marker):
    source = next(source for source in sources if marker in source)
    if marker == "from sklearn.model_selection":
        scope.update(sample=pd.read_parquet(root / "cw2_joint_samples.parquet"),
                     features=["S1_MW", "S2_MW", "S3_MW", "P_pu", "N"], target="f_nadir_hz")
    if marker == "accuracy_models = {}":
        start = source.index("    for name,shape in ")
        end = source.index("\n", start)
        source = source[:start] + '    for name,shape in [("64+64",(64,64))]:' + source[end:]
    exec(source.split("for row in sample.sample")[0].split("points=np.arange")[0], scope)


for marker in ("def storage_bound(", "def response_batch(", "def frequency_metrics("):
    load_cell(marker)
exec(next(line for source in sources for line in source.splitlines() if line.startswith("costs = ")), scope)
costs = scope["costs"]
unit_cost = scope["unit_cost"]
matrix = np.vstack([np.ones(3), -scope["m_i"], -(scope["d_i"] + scope["k_i"])])


def solve_analytic_local(load, counts=None, seed=None):
    candidates = []
    for count in (range(scope["storage_bound"](load) + 1) if counts is None else counts):
        rhs = np.array([240, -10000 * load, -5000 * abs(load - .01 * count) / .3])
        lower = linprog(costs, A_ub=matrix, b_ub=rhs, method="highs")
        if lower.success:
            candidates.append((lower.fun + unit_cost * count, count, lower.x, rhs))
    best = None
    best_cost = np.inf
    for lower_cost, count, capacity, rhs in sorted(candidates, key=lambda item: item[0]):
        if lower_cost >= best_cost - 1e-8:
            break

        def margin(capacity):
            installed_inertia = scope["m_i"] @ capacity
            minimum_inertia = 10000 * load
            if installed_inertia <= 0:
                return -1.
            if installed_inertia < minimum_inertia:
                scaled_capacity = capacity * (minimum_inertia / installed_inertia)
                scaled_drop = scope["response_batch"](scaled_capacity[None, :], load, count)[0][0]
                return .5 * installed_inertia / minimum_inertia - scaled_drop
            return .5 - scope["response_batch"](capacity[None, :], load, count)[0][0]

        for initial in ((seed,) if seed is not None else ()) + (capacity, capacity + (240 - capacity.sum()) / 3):
            result = minimize(lambda installed: costs @ installed, initial,
                              jac=lambda installed: costs, bounds=[(0, 240)] * 3,
                              method="SLSQP",
                              constraints=[{"type": "ineq", "fun": lambda installed: rhs - matrix @ installed},
                                           {"type": "ineq", "fun": margin}],
                              options={"ftol": 1e-9, "maxiter": 80})
            price = result.fun + unit_cost * count
            if result.success and price < best_cost and margin(result.x) >= -1e-7 and np.min(rhs - matrix @ result.x) >= -1e-7:
                best_cost, best = price, (result.x, count)
    return best


def solve_analytic(load, tolerance=.01, node_budget=2000, strategy="enumerated", audit=None):
    if 10000 * load > 240 * max(scope["m_i"]) + 1e-8:
        return None, np.inf, "infeasible_by_inertia"
    started = perf_counter()
    max_count = scope["storage_bound"](load)
    joint_costs = np.r_[costs, unit_cost]
    steady = 5000 / .3
    joint_matrix = np.vstack([np.c_[matrix, [0, 0, -steady * .01]],
                              np.r_[matrix[-1], steady * .01]])
    joint_rhs = np.array([240, -10000 * load, -steady * load, steady * load])

    def joint_lp(first, last):
        result = linprog(joint_costs, A_ub=joint_matrix, b_ub=joint_rhs,
                         bounds=[(0, 240)] * 3 + [(first, last)], method="highs")
        if not result.success and result.status != 2:
            raise RuntimeError(result.message)
        return result

    candidate = solve_analytic_local(load) if strategy == "enumerated" else None
    if strategy == "warm":
        relaxed_lp = joint_lp(0, max_count)
        if relaxed_lp.success:
            def margin(decision):
                capacity, count = decision[:3], decision[3]
                installed_inertia = scope["m_i"] @ capacity
                if installed_inertia <= 0:
                    return -1.
                scale = max(1., 10000 * load / installed_inertia)
                return .5 / scale - scope["response_batch"]((scale * capacity)[None, :], load, count)[0][0]

            relaxed = minimize(lambda decision: joint_costs @ decision, relaxed_lp.x,
                               jac=lambda decision: joint_costs, method="SLSQP",
                               bounds=[(0, 240)] * 3 + [(0, max_count)],
                               constraints=[{"type": "ineq", "fun": lambda decision: joint_rhs - joint_matrix @ decision},
                                            {"type": "ineq", "fun": margin}],
                               options={"ftol": 1e-9, "maxiter": 80})
            seed = relaxed.x if relaxed.success else relaxed_lp.x
            counts = sorted({int(np.floor(seed[3])), int(np.ceil(seed[3]))} & set(range(max_count + 1)))
            candidate = solve_analytic_local(load, counts, seed[:3])
    upper = costs @ candidate[0] + unit_cost * candidate[1] if candidate else np.inf
    first_feasible = perf_counter() - started if candidate else np.nan
    queue = []
    serial = 0
    minimum_inertia = 100 * load
    duration = scope["delivery_time"]
    inertia = scope["m_i"] / 100
    damping = scope["d_i"] / 100
    gains = scope["k_i"] / (100 * scope["tau_i"])

    def enqueue(count, lo, hi):
        nonlocal serial, candidate, upper, first_feasible
        rhs = [240, -10000 * load, -5000 * abs(load - .01 * count) / .3]
        bounds = list(zip(lo, hi))
        result = linprog(costs, A_ub=matrix, b_ub=rhs, bounds=bounds, method="highs")
        if result.status == 2:
            return
        if not result.success:
            raise RuntimeError(result.message)
        lower = result.fun + unit_cost * count - 1e-8
        if lower >= upper:
            return
        drop, times = scope["response_batch"](result.x[None, :], load, count)
        if drop[0] <= .5 + 1e-9:
            if lower + 1e-8 < upper:
                candidate, upper = (result.x, count), lower + 1e-8
                if np.isnan(first_feasible):
                    first_feasible = perf_counter() - started
            if lower < upper:
                serial += 1
                heappush(queue, (lower, serial, count, lo, hi))
        else:
            time = times[0]
            if np.isfinite(time) and time > 0:
                # Bound every trajectory in the box at the LP solution's violating time.
                center = result.x
                radius = np.maximum(center - lo, hi - center)
                min_mass = max(inertia @ lo, minimum_inertia)
                mass = inertia @ center
                inv_error = (inertia @ radius) / (mass * min_mass)
                reserve = .01 * count
                system = np.zeros((6, 6))
                system[0, :5] = np.r_[-damping @ center, 1, 1, 1, 1] / mass
                system[1:4, 0] = -gains * center
                system[1:4, 1:4] = -np.diag(1 / scope["tau_i"])
                system[4, 5] = 1
                delta = np.zeros_like(system)
                delta[0, 0] = damping @ radius / min_mass + (damping @ center) * inv_error
                delta[0, 1:5] = inv_error
                delta[1:4, 0] = gains * radius
                comparison = abs(system) + delta
                np.fill_diagonal(comparison, system.diagonal() + delta.diagonal())
                augmented = np.block([[system, np.zeros_like(system)], [delta, comparison]])
                state = np.zeros(12)
                state[4:6] = [-load, reserve / duration]
                state = expm(augmented * min(time, duration)) @ state
                if time > duration:
                    state[5] = 0
                    state = expm(augmented * (time - duration)) @ state
                # The lower block propagates a bound using |nominal state|.
                # Signed nominal forcing cannot bound the error, so use an absolute envelope below.
                envelope = abs(system)
                np.fill_diagonal(envelope, system.diagonal())
                augmented[:6, :6] = envelope
                bound = np.zeros(12)
                bound[4:6] = [load, reserve / duration]
                bound = expm(augmented * min(time, duration)) @ bound
                if time > duration:
                    bound[5] = 0
                    bound = expm(augmented * (time - duration)) @ bound
                if np.isfinite(bound[6]) and 50 * (state[0] + bound[6]) < -.5 - 1e-8:
                    return
        serial += 1
        heappush(queue, (lower, serial, count, lo, hi))

    def enqueue_fixed(count):
        nonlocal candidate, upper, first_feasible
        if strategy != "enumerated":
            local = solve_analytic_local(load, [count])
            if local and costs @ local[0] + unit_cost * count < upper:
                candidate, upper = local, costs @ local[0] + unit_cost * count
                if np.isnan(first_feasible):
                    first_feasible = perf_counter() - started
        hi = np.minimum(240, (upper - unit_cost * count) / costs) if np.isfinite(upper) else np.full(3, 240.)
        if min(hi) >= 0:
            enqueue(count, np.zeros(3), hi)

    def enqueue_interval(first, last):
        nonlocal serial
        result = joint_lp(first, last)
        if result.success and result.fun - 1e-8 < upper:
            serial += 1
            heappush(queue, (result.fun - 1e-8, serial, (first, last), None, None))

    if strategy == "enumerated":
        for count in range(max_count + 1):
            enqueue_fixed(count)
    else:
        enqueue_interval(0, max_count)
    nodes = 0
    for _ in range(node_budget):
        if not queue or queue[0][0] >= upper - tolerance:
            break
        lower, serial_number, count, lo, hi = heappop(queue)
        nodes += 1
        if isinstance(count, tuple):
            first, last = count
            if first == last:
                enqueue_fixed(first)
            else:
                middle = (first + last) // 2
                enqueue_interval(first, middle)
                enqueue_interval(middle + 1, last)
            continue
        index = int(np.argmax(hi - lo))
        midpoint = (lo[index] + hi[index]) / 2
        if midpoint <= lo[index] or midpoint >= hi[index]:
            heappush(queue, (lower, serial_number, count, lo, hi))
            break
        child_hi, child_lo = hi.copy(), lo.copy()
        child_hi[index] = child_lo[index] = midpoint
        enqueue(count, lo, child_hi)
        enqueue(count, child_lo, hi)
    lower = min(upper, queue[0][0]) if queue else upper
    status = ("gap_met" if upper - lower <= tolerance else "budget_or_resolution") if candidate else (
        "budget_or_resolution" if queue else "infeasible_by_bounds"
    )
    if audit is not None:
        audit.update(nodes=nodes, first_feasible_seconds=first_feasible)
    return candidate, lower, status


def main():
    for marker in ("from sklearn.model_selection", "accuracy_models = {}", "def solve_capacity("):
        load_cell(marker)
    network = scope["models"]["64+64"]
    scope["solve_capacity"](.10, network)
    records = []
    for load in np.arange(1, 101) / 100:
        record = {"P_pu": load}
        for label, solve in (("analytic", solve_analytic), ("milp", lambda load: scope["solve_capacity"](load, network))):
            start = perf_counter()
            result = solve(load)
            if label == "analytic":
                result, record["analytic_lower_bound"], record["analytic_status"] = result
            record[f"{label}_seconds"] = perf_counter() - start
            record[f"{label}_cost"] = np.nan
            record[f"{label}_dynamic_ok"] = False
            for metric in ("rocof_abs", "steady_abs", "nadir_hz"):
                record[f"{label}_{metric}"] = np.nan
            for variable in ("S1_MW", "S2_MW", "S3_MW", "N"):
                record[f"{label}_{variable}"] = np.nan
            if result is not None:
                capacity, count = result
                slope, steady, nadir = scope["frequency_metrics"](capacity, load, count)
                record[f"{label}_cost"] = costs @ capacity + unit_cost * count
                for variable, value in zip(("S1_MW", "S2_MW", "S3_MW", "N"), (*capacity, count)):
                    record[f"{label}_{variable}"] = value
                record[f"{label}_rocof_abs"] = abs(slope)
                record[f"{label}_steady_abs"] = abs(steady)
                record[f"{label}_nadir_hz"] = 50 + nadir
                record[f"{label}_dynamic_ok"] = abs(slope) <= .5000001 and abs(steady) <= .3000001 and nadir >= -.5000001
                if label == "analytic":
                    record["analytic_gap"] = record["analytic_cost"] - record["analytic_lower_bound"]
                    if not record["analytic_dynamic_ok"]:
                        record["analytic_status"] = "dynamic_check_failed"
        records.append(record)
    add_static_and_plot(pd.DataFrame(records))


def add_static_and_plot(data):
    for row, load in data.P_pu.items():
        start = perf_counter()
        best = np.inf
        solution = None
        for count in range(scope["storage_bound"](load) + 1):
            rhs = [240, -10000 * load, -5000 * abs(load - .01 * count) / .3]
            candidate = linprog(costs, A_ub=matrix, b_ub=rhs, method="highs")
            if candidate.success and candidate.fun + unit_cost * count < best:
                best = candidate.fun + unit_cost * count
                solution = candidate.x, count
        data.loc[row, "static_seconds"] = perf_counter() - start
        data.loc[row, "static_cost"] = best if np.isfinite(best) else np.nan
        if solution is not None:
            capacity, count = solution
            slope, steady, nadir = scope["frequency_metrics"](capacity, load, count)
            data.loc[row, ["static_S1_MW", "static_S2_MW", "static_S3_MW", "static_N"]] = [*capacity, count]
            data.loc[row, ["static_rocof_abs", "static_steady_abs", "static_nadir_hz"]] = [abs(slope), abs(steady), 50 + nadir]
    data.to_csv(root / "analytic_milp_comparison.csv", index=False)
    fig, cost_axis = plt.subplots(figsize=(9, 4.8))
    time_axis = cost_axis.twinx()
    lines = []
    for label, color, name in (("analytic", "tab:blue", "Analytic B&B"), ("milp", "tab:orange", "MILP"),
                               ("static", "tab:green", "No nadir constraint")):
        lines += cost_axis.plot(data.P_pu, data[f"{label}_cost"], color=color, label=f"{name} cost")
        lines += time_axis.plot(data.P_pu, data[f"{label}_seconds"].where(data[f"{label}_cost"].notna()), "--", color=color, label=f"{name} time")
    shown = data.loc[data[["analytic_cost", "milp_cost", "static_cost"]].notna().any(axis=1), "P_pu"]
    cost_axis.set_xlim(shown.min(), shown.max())
    cost_axis.set(xlabel="Load step P (p.u.)", ylabel="Annual cost (million yuan/year)")
    time_axis.set(ylabel="Solve time (s, log scale)", yscale="log")
    cost_axis.legend(handles=lines)
    unresolved = data.analytic_cost.notna() & data.analytic_status.ne("gap_met")
    if unresolved.any():
        cost_axis.scatter(data.P_pu[unresolved], data.analytic_cost[unresolved],
                          facecolors="none", edgecolors="tab:blue", label="Analytic gap unresolved")
    cost_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)
    cost_axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(root / "analytic_milp_comparison.png", dpi=160)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    metric_specs = [
        ("rocof_abs", "Initial |RoCoF| (Hz/s)", .50),
        ("steady_abs", "Steady-state |deviation| (Hz)", .30),
        ("nadir_hz", "Frequency nadir (Hz)", 49.50),
    ]
    for axis, (metric, ylabel, limit) in zip(axes, metric_specs):
        axis.plot(data.P_pu, data[f"analytic_{metric}"], label="Analytic B&B", color="tab:blue")
        axis.plot(data.P_pu, data[f"milp_{metric}"], label="ReLU + MILP", color="tab:orange", marker=".", markersize=3)
        axis.plot(data.P_pu, data[f"static_{metric}"], label="Static planning", color="tab:green", marker="x", markersize=3)
        axis.axhline(limit, linestyle="--", color="black", label=f"Limit {limit:.2f}")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
        axis.legend(loc="best")
    shown = data.loc[data[["analytic_cost", "milp_cost", "static_cost"]].notna().any(axis=1), "P_pu"]
    axes[0].set_ylim(0.49, 0.51)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[-1].set(xlabel="Load step P (p.u.)", xlim=(shown.min(), shown.max()))
    fig.tight_layout()
    fig.savefig(root / "analytic_milp_frequency.png", dpi=160)
    print(data.loc[np.isclose(data.P_pu, .10)].to_string(index=False))
    print(data[["analytic_dynamic_ok", "milp_dynamic_ok"]].sum())


if __name__ == "__main__":
    main()

