#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sko.PSO import PSO


SEED = 42
POP_SIZE = 30
MAX_ITER = 100
BOUNDS = (-5.0, 5.0)

# Each line defines a half-plane: a * x + b * y <= c
CONSTRAINTS = [
	(1.0, 1.0, 4.0),
	(-1.0, 1.0, 4.0),
	(1.0, -1.0, 4.0),
	(-1.0, -1.0, 4.0),
]
PENALTY_WEIGHT = 1e10
GLOBAL_CENTER = np.array([1.0, 1.0])
GLOBAL_DEPTH = 3.0e5
GLOBAL_SIGMA = 0.5
LOCAL_CENTER = np.array([-1.0, -1.0])
LOCAL_DEPTH = 2.9e5
LOCAL_SIGMA = 0.8
CONTOUR_GRID = 200
REWARD_KEEP_PERCENTILE = 70
REWARD_CONTOUR_LEVELS = 12


def _constraint_violation(xy):
	x, y = float(xy[0]), float(xy[1])
	violation = 0.0
	for a, b, c in CONSTRAINTS:
		violation += max(0.0, a * x + b * y - c) ** 2
	return violation


def _is_feasible(xy):
	for a, b, c in CONSTRAINTS:
		if a * xy[0] + b * xy[1] > c:
			return False
	return True


def _objective_single(xy):
	penalty = _constraint_violation(xy)
	local_dx = float(xy[0] - LOCAL_CENTER[0])
	local_dy = float(xy[1] - LOCAL_CENTER[1])
	global_dx = float(xy[0] - GLOBAL_CENTER[0])
	global_dy = float(xy[1] - GLOBAL_CENTER[1])

	local_well = -LOCAL_DEPTH * np.exp(
		-0.5 * ((local_dx / LOCAL_SIGMA) ** 2 + (local_dy / LOCAL_SIGMA) ** 2)
	)
	global_well = -GLOBAL_DEPTH * np.exp(
		-0.5 * ((global_dx / GLOBAL_SIGMA) ** 2 + (global_dy / GLOBAL_SIGMA) ** 2)
	)
	return global_well + local_well + PENALTY_WEIGHT * penalty

def objective(x):
	x = np.asarray(x)
	if x.ndim == 1:
		return _objective_single(x)
	return np.array([_objective_single(row) for row in x])


def run_pso_with_trace():
	np.random.seed(SEED)
	lb = [BOUNDS[0], BOUNDS[0]]
	ub = [BOUNDS[1], BOUNDS[1]]

	pso = PSO(
		func=objective,
		dim=2,
		pop=POP_SIZE,
		max_iter=MAX_ITER,
		w=0.7,
		c1=1.5,
		c2=1.5,
		lb=lb,
		ub=ub,
		verbose=True
	)

	positions = [pso.X.copy()]
	for _ in range(MAX_ITER):
		pso.update_V()
		pso.update_X()
		pso.cal_y()
		pso.update_pbest()
		pso.update_gbest()
		positions.append(pso.X.copy())

	return positions, pso.gbest_x, pso.gbest_y


def _plot_constraints(ax, xlim, ylim):
	xs = np.linspace(xlim[0], xlim[1], 400)
	for a, b, c in CONSTRAINTS:
		if abs(b) < 1e-9:
			x = c / a
			ax.plot([x, x], [ylim[0], ylim[1]], color="black", linewidth=1.0)
		else:
			ys = (c - a * xs) / b
			ax.plot(xs, ys, color="black", linewidth=1.0)


def _plot_reward_contours(ax, xlim, ylim):
	xs = np.linspace(xlim[0], xlim[1], CONTOUR_GRID)
	ys = np.linspace(ylim[0], ylim[1], CONTOUR_GRID)
	grid_x, grid_y = np.meshgrid(xs, ys)
	points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

	feasible_mask = np.array([_is_feasible(p) for p in points])
	reward = -objective(points)
	reward[~feasible_mask] = np.nan

	finite = reward[np.isfinite(reward)]
	if finite.size == 0:
		return
	threshold = np.percentile(finite, REWARD_KEEP_PERCENTILE)
	reward[reward < threshold] = np.nan

	contours = ax.contourf(
		grid_x,
		grid_y,
		reward.reshape(grid_x.shape),
		levels=REWARD_CONTOUR_LEVELS,
		cmap="magma",
		alpha=0.65,
	)
	fig = ax.figure
	fig.colorbar(contours, ax=ax, shrink=0.75, label="reward")


def plot_positions(positions):
	fig, ax = plt.subplots(figsize=(7.5, 7.5))
	xlim = (BOUNDS[0], BOUNDS[1])
	ylim = (BOUNDS[0], BOUNDS[1])

	_plot_reward_contours(ax, xlim, ylim)
	_plot_constraints(ax, xlim, ylim)
	ax.scatter([0.0], [0.0], s=60, c="red", marker="*", label="origin")

	colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(positions)))
	for idx, (cloud, color) in enumerate(zip(positions, colors)):
		ax.scatter(
			cloud[:, 0],
			cloud[:, 1],
			s=18,
			alpha=0.65,
			color=color,
			label=f"iter {idx}",
		)

	ax.set_title("PSO particles with hard-line penalties")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_aspect("equal", adjustable="box")
	# ax.legend(loc="upper right", fontsize=8, ncol=2)
	ax.grid(True, linestyle="--", alpha=0.3)
	plt.tight_layout()
	plt.show()


def main():
	positions, best_x, best_y = run_pso_with_trace()
	print(f"Best x: {best_x}")
	print(f"Best fitness: {best_y}")
	plot_positions(positions)


if __name__ == "__main__":
	main()

