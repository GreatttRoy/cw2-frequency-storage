"""用图 6 的响应基底生成固定视角的储能规划动画。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


directory = Path(__file__).resolve().parent
notebook = json.loads((directory / "cw2_Q3加入储能和规划.ipynb").read_text())
sources = ["".join(cell["source"]) for cell in notebook["cells"]]
parameters = next(source for source in sources if "def storage_bound(" in source)
figure_source = next(source for source in sources if 'id="efr-range"' in source)
scope = {}
exec(parameters, scope)
exec(figure_source.split("payload=", 1)[0], scope)
coordinates = scope["xyz"][scope["inside"]]
basis = scope["basis"]

load_steps = np.rint(np.linspace(1, 100, 40)).astype(int) / 100
fixed_load = .10
count_steps = np.arange(scope["storage_bound"](fixed_load) + 1)
loads = np.r_[load_steps, np.full(len(count_steps), fixed_load)]
counts = np.r_[np.full(len(load_steps), 7), count_steps]
assert all(count <= scope["storage_bound"](load) for load, count in zip(loads, counts))
nadirs = [
    50 + (load * basis[:, :, 0] + .01 * count * basis[:, :, 1]).min(axis=1)
    for load, count in zip(loads, counts)
]

figure = plt.figure(figsize=(8, 5.6))
axis = figure.add_subplot(projection="3d")
axis.set(xlabel="S1 (MW)", ylabel="S2 (MW)", zlabel="S3 (MW)",
         xlim=(0, 240), ylim=(0, 240), zlim=(0, 240))
axis.set_box_aspect((1, 1, 1))
axis.view_init(elev=22, azim=-110)
points = axis.scatter(*coordinates.T, c=nadirs[0], cmap="viridis",
                      vmin=49, vmax=50, s=15, depthshade=False)
figure.colorbar(points, ax=axis, label="Frequency nadir (Hz)", shrink=.72, pad=.12)
title = figure.text(.48, .90, "", ha="center")
figure.text(.48, .03, "Figure 6-3. Separate load-step and storage-count sweeps", ha="center")


def draw_frame(frame):
    points.set_array(nadirs[frame])
    phase = "1. Increase P at fixed N" if frame < len(load_steps) else "2. Increase N at fixed P"
    title.set_text(f"{phase}\nP = {loads[frame]:.2f} p.u.     N = {counts[frame]} units")


frames = ([0] * 5 + list(range(len(load_steps))) + [len(load_steps) - 1] * 5
          + [len(load_steps)] * 5 + list(range(len(load_steps), len(loads))) + [len(loads) - 1] * 8)
animation = FuncAnimation(figure, draw_frame, frames=frames)
animation.save(directory / "figure6_p_n.gif", writer=PillowWriter(fps=6), dpi=100)
