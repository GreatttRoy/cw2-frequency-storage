import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize

notebook = Path(__file__).with_name("cw2_Q3加入储能和规划.ipynb")
cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]]
scope = {"np": np}
parameters = next(source for source in cells if "def storage_bound(" in source)
exec(parameters[parameters.index("m_i ="):], scope)
exec(next(source for source in cells if source.startswith("def response_batch")), scope)
validation = next(source for source in cells if "def frequency_metrics(" in source)
exec(validation.split("for row in sample.sample")[0], scope)

load = .10
costs = np.array([.75, 1.05, 1.65])
inertia = scope["m_i"]
gain = scope["d_i"] + scope["k_i"]
matrix = np.vstack([np.ones(3), -inertia, -gain])
incumbent = (costs @ [1000/14, 0, 0] + 7*scope["unit_cost"], np.array([1000/14, 0, 0]), 7)
assert scope["frequency_metrics"](incumbent[1], load, 7)[2] >= -.5
start = perf_counter()
attempts = 0
for count in range(scope["storage_bound"](load)+1):
    rhs = np.array([240, -10000*load, -5000*abs(load-.01*count)/.3])
    fixed_cost = count*scope["unit_cost"]
    lower = linprog(costs, A_ub=matrix, b_ub=rhs, method="highs")
    if not lower.success or lower.fun + fixed_cost >= incumbent[0]:
        continue
    attempts += 1
    def margin(capacity):
        return .5-scope["response_batch"](capacity[None,:], load, count)[0][0]
    for initial in (lower.x, lower.x + (240-lower.x.sum())/3):
        result = minimize(lambda capacity: costs@capacity, initial, jac=lambda capacity: costs,
                          bounds=[(0,240)]*3, method="SLSQP",
                          constraints=[{"type":"ineq","fun":lambda capacity: rhs-matrix@capacity},
                                       {"type":"ineq","fun":margin}],
                          options={"ftol":1e-9,"maxiter":80})
        if result.success and result.fun+fixed_cost < incumbent[0]:
            metrics = scope["frequency_metrics"](result.x, load, count)
            if np.min(rhs-matrix@result.x)>=-1e-7 and metrics[2]>=-.5000001:
                incumbent = (result.fun+fixed_cost, result.x, count)
elapsed = perf_counter()-start
cost, capacity, count = incumbent
print("S (MW)", capacity, "N", count, "annual cost", cost)
print("RoCoF, steady deviation, nadir deviation",scope["frequency_metrics"](capacity,load,count))
print("NLP subproblems", attempts, "seconds", elapsed)

