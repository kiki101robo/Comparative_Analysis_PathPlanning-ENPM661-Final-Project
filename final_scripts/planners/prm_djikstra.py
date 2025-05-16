import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
import time
from matplotlib.animation import FuncAnimation
import heapq
import sys
import os
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from map.mapenv0 import MapEnv
from userinput import UserInput
from utils import compute_path_length, compute_path_jerkiness


class PRMPlanner:
    def __init__(
        self, map_env, canvas, start, goal, rpm1, rpm2, num_samples=1000, k=10
    ):
        self.env = map_env
        self.canvas = canvas
        self.start = start
        self.goal = goal
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.num_samples = num_samples
        self.k = k
        self.nodes = []
        self.edges = {}

    def motion_sim(self, start, move, dt=0.8):
        x, y, theta = start
        theta_rad = math.radians(theta)
        rad_rpm1 = (2 * math.pi / 60) * move[0]
        rad_rpm2 = (2 * math.pi / 60) * move[1]
        lin_vel = 3.3 * (rad_rpm1 + rad_rpm2) / 2
        ang_vel = 3.3 * (rad_rpm1 - rad_rpm2) / 2.87
        x_new = x + lin_vel * math.cos(theta_rad) * dt
        y_new = y + lin_vel * math.sin(theta_rad) * dt
        theta_new = (theta_rad + ang_vel * dt + math.pi) % (2 * math.pi) - math.pi
        return x_new, y_new, math.degrees(theta_new)

    def sample_free_space(self):
        h, w = self.canvas.shape[:2]
        samples = []
        while len(samples) < self.num_samples:
            x = np.random.uniform(0, w)
            y = np.random.uniform(0, h)
            if not self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                samples.append((x, y))
        return samples

    def simulate_dd_path(self, path):
        sim_path = []
        for i in range(len(path) - 1):
            x, y = path[i]
            x_goal, y_goal = path[i + 1]
            best_move = None
            best_result = None
            min_dist = float("inf")
            for move in self.get_actions():
                x_new, y_new, theta_new = self.motion_sim((x, y, 0), move)
                dist = np.linalg.norm([x_new - x_goal, y_new - y_goal])
                if dist < min_dist and self.is_motion_collision_free(
                    (x, y), (x_new, y_new)
                ):
                    min_dist = dist
                    best_result = (x_new, y_new, theta_new, move[0], move[1])
            if best_result:
                sim_path.append(best_result)
        return sim_path

    def build_roadmap(self):
        self.nodes = self.sample_free_space()
        self.nodes += [(270, 50), (270, 150), (270, 250)]
        self.nodes.append((self.start[0], self.start[1]))
        self.nodes.append((self.goal[0], self.goal[1]))
        kdtree = KDTree(self.nodes)
        self.edges = {i: [] for i in range(len(self.nodes))}
        pairs_to_check = []
        for i, node in enumerate(self.nodes):
            _, idxs = kdtree.query(node, k=self.k + 1)
            for j in idxs:
                if j <= i:
                    continue
                pairs_to_check.append((i, j))

        def validate_edge(i, j):
            if self.collision_free(self.nodes[i], self.nodes[j]):
                dist = np.linalg.norm(np.array(self.nodes[i]) - np.array(self.nodes[j]))
                return i, j, dist
            return None

        results = Parallel(n_jobs=-1)(
            delayed(validate_edge)(i, j) for i, j in pairs_to_check
        )
        for result in results:
            if result:
                i, j, dist = result
                self.edges[i].append((j, dist))
                self.edges[j].append((i, dist))
        return self.edges

    def get_actions(self):
        return [
            [0, self.rpm1],
            [0, self.rpm2],
            [self.rpm1, 0],
            [self.rpm2, 0],
            [self.rpm1, self.rpm2],
            [self.rpm2, self.rpm1],
            [self.rpm1, self.rpm1],
            [self.rpm2, self.rpm2],
        ]

    def is_motion_collision_free(self, p1, p2, steps=10):
        for i in range(steps + 1):
            u = i / steps
            x = p1[0] * (1 - u) + p2[0] * u
            y = p1[1] * (1 - u) + p2[1] * u
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return False
        return True

    def collision_free(self, p1, p2):
        return self.is_motion_collision_free(p1, p2)

    def dijkstra(self):
        start_idx = len(self.nodes) - 2
        goal_idx = len(self.nodes) - 1
        dist = {i: float("inf") for i in range(len(self.nodes))}
        prev = {i: None for i in range(len(self.nodes))}
        dist[start_idx] = 0
        pq = [(0, start_idx)]

        while pq:
            _, u = heapq.heappop(pq)
            if u == goal_idx:
                break
            for v, cost in self.edges[u]:
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        path = []
        current = goal_idx
        while current is not None:
            path.append(self.nodes[current])
            current = prev[current]
        path.reverse()
        return path if path and path[0] == (self.start[0], self.start[1]) else None

    def plan(self, visualize=True):
        self.build_roadmap()
        if visualize:
            self.visualize_prm()
        return self.dijkstra()

    def visualize_prm(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.canvas.copy(), cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        xs, ys = zip(*self.nodes)
        ax.scatter(xs, ys, s=5, c="cyan", alpha=0.6)
        for i in self.edges:
            for j, _ in self.edges[i]:
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
                ax.plot([x1, x2], [y1, y2], "gray", linewidth=0.5, alpha=0.5)
        ax.scatter([self.start[0]], [self.start[1]], c="green", s=100)
        ax.scatter([self.goal[0]], [self.goal[1]], c="red", s=100)
        plt.title("PRM Roadmap")
        plt.grid(True)
        plt.show()


# ----------------------------------------------------------------------
#  Batch‑mode entry point used by evaluate_all.py
# ----------------------------------------------------------------------
def run_planner(
    env,
    start,
    goal,
    rpm1,
    rpm2,
    *,
    num_samples: int = 1000,
    k: int = 10,
    visualize: bool = False,
):
    """
    External wrapper so the evaluator can call PRM‑Dijkstra exactly like the
    other planners.

    Returns
    -------
    list[tuple] | None
        Successful path as [(x, y), ...]   — or None on failure.
    """
    # --------------------------------------------------------------
    # Obtain or (re‑)create a *proper* NumPy image for sampling
    # --------------------------------------------------------------
    canvas = getattr(env, "canvas", None)

    ok = isinstance(canvas, np.ndarray) and canvas.ndim in (2, 3)
    if not ok:
        # Either missing or invalid → make a fresh one
        canvas_out = env.create_canvas(540, 300)
        # MapEnv may return (image, inflated_obs)  or just image
        canvas = canvas_out[0] if isinstance(canvas_out, tuple) else canvas_out

    # Store the clean image back so everyone else sees the right object
    env.canvas = canvas

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    planner = PRMPlanner(
        env,
        canvas,
        start,
        goal,
        rpm1,
        rpm2,
        num_samples=num_samples,
        k=k,
    )

    raw_path = planner.plan(visualize=visualize)

    if not raw_path:
        return None

    # convert to floats so downstream utilities don’t choke on ints
    return [(float(x), float(y)) for x, y in raw_path]


class Visualizer:
    """
    Manages visualization of exploration, path animation, and final path using matplotlib.
    """

    def __init__(self, map, start, goal, scale=1.0):
        self.map = map.copy()
        self.start = start
        self.goal = goal
        self.scale = scale
        self.explored_nodes = []
        self.path = []

    def record_path(self, path):
        """
        Stores the final path to be visualized.
        """
        self.path = path

    def animate_explore(self):
        """
        Generates an animated scatter plot of explored points using matplotlib.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter(
            [self.start[0]],
            [self.start[1]],
            c="green",
            s=100,
            label="Start",
            marker="o",
        )
        ax.scatter(
            [self.goal[0]], [self.goal[1]], c="red", s=100, label="Goal", marker="o"
        )
        plt.title("Exploration Animation")
        plt.legend()
        plt.draw()
        plt.show(block=True)
        plt.close()

    def animate_path(self):
        """
        Generates a path-following animation with a moving point.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter(
            [self.start[0]],
            [self.start[1]],
            c="green",
            s=100,
            label="Start",
            marker="o",
        )
        ax.scatter(
            [self.goal[0]], [self.goal[1]], c="red", s=100, label="Goal", marker="o"
        )
        path_x, path_y = [pt[0] for pt in self.path], [pt[1] for pt in self.path]
        ax.plot(path_x, path_y, c="red", linewidth=3, label="Path")
        plt.title("Path Animation")
        plt.legend()
        (line,) = ax.plot([], [], "bo", ms=10)

        def update(frame):
            line.set_data([path_x[frame]], [path_y[frame]])
            return (line,)

        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(
            fig, update, frames=range(len(self.path)), interval=50, blit=True
        )
        plt.show()
        plt.show(block=True)
        plt.close()

    def draw_path(self):
        """
        Displays the final path overlayed on the map.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter(
            [self.start[0]],
            [self.start[1]],
            c="green",
            s=100,
            label="Start",
            marker="o",
        )
        ax.scatter(
            [self.goal[0]], [self.goal[1]], c="red", s=100, label="Goal", marker="o"
        )
        if self.path:
            path_x, path_y = zip(*[(pt[0], pt[1]) for pt in self.path])
            ax.plot(path_x, path_y, c="red", linewidth=3, label="Path")
        plt.title("Final Path")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.show(block=True)
        plt.close()


# ----------------------------------------------------------------------
# Batch‑mode entry point expected by evaluate_all.py
# Signature: run_planner(env, start, goal, rpm1, rpm2, **kwargs)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    height, width = 300, 540
    input_handler = UserInput()
    clearance = input_handler.get_clearance()
    robot_radius = 2.2
    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    rpm1, rpm2 = input_handler.get_rpm()
    start, goal = input_handler.get_start_goal(env, width, height)

    print(f"\nUsing RPMs: ({rpm1}, {rpm2})")
    print(f"Start: {start}, Goal: {goal}")

    planner = PRMPlanner(env, canvas, start, goal, rpm1, rpm2, num_samples=1000, k=10)
    start_time = time.time()
    path = planner.plan()
    end_time = time.time()
    planner = PRMPlanner(env, canvas, start, goal, rpm1, rpm2, num_samples=1000, k=10)
    vis = Visualizer(canvas, start, goal)

    start_time = time.time()
    path = planner.plan()
    end_time = time.time()

    if path:
        sim_path = planner.simulate_dd_path(path)
        if sim_path:
            ...
            vis.record_path(sim_path)
            vis.animate_path()
            vis.draw_path()
    else:
        print("\nNo valid path found using PRM.")
