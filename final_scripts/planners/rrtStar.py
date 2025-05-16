import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import math
import heapq as hp
import time
import sys
import os

# top‑level imports
from scipy.spatial import KDTree  # NEW


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from map.mapenv1 import MapEnv
from userinput import UserInput
from utils import compute_path_length, compute_path_jerkiness


class Node:
    def __init__(self, x, y, theta, rpm=(0, 0), parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.rpm = rpm
        self.parent = parent

    def position(self):
        return (self.x, self.y)

    def state(self):
        return (self.x, self.y, self.theta)


# ─────────────────────────────────────────────────────────────────────────
#  RRT*  (full replacement for the old RRTPlanner class)
# ─────────────────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree  # <- keep this with the other imports


class RRTPlanner:
    """
    γ‑RRT* with KD‑tree acceleration.
    Keeps the original API: plan(width, height, vis) → list[(x,y)] or None
    """

    def __init__(
        self,
        rpm1,
        rpm2,
        start,
        goal,
        env,
        max_iter=5000,
        goal_sample_rate=0.35,  # ↑ goal bias
        rebuild_kd_freq=200,
    ):
        # ----- user/lab data -----
        self.rpm1, self.rpm2 = rpm1, rpm2
        self.start = Node(*start)
        self.start.cost = 0
        self.goal = Node(*goal)
        self.env = env

        # ----- planner params -----
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.threshold = 35.0  # goal radius (px)
        self.gamma_rrt = 30.0  # rewiring radius constant
        self.dim = 2  # workspace dimension
        self.rebuild_kd_freq = rebuild_kd_freq

        # ----- data structures -----
        self.tree = [self.start]  # list[Node]
        self.edges = []  # for visualiser
        self._coords = [(self.start.x, self.start.y)]
        self._kd = KDTree(self._coords)

        self.path = []  # final path (x,y)

    # ───────────────────────────────── helpers ──────────────────────────
    def _dist(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def nearest(self, _nodes, target):
        """KD‑tree nearest neighbour."""
        _, idx = self._kd.query([target.x, target.y])
        return self.tree[idx]

    def near(self, node, radius):
        idxs = self._kd.query_ball_point([node.x, node.y], radius)
        return [self.tree[i] for i in idxs]

    # sample, moves & motion are unchanged -------------------------------
    def move_set(self):
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

    def sample_free(self, w, h):
        if np.random.rand() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y, self.goal.theta)
        while True:
            x, y = np.random.uniform(0, w), np.random.uniform(0, h)
            if not self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return Node(x, y, np.random.uniform(-180, 180))

    def simulate_motion(self, node, move):
        # identical except for coarser dt / shorter arc
        rad1 = (2 * math.pi / 60) * move[0]
        rad2 = (2 * math.pi / 60) * move[1]
        lin = 3.3 * (rad1 + rad2) / 2
        ang = 3.3 * (rad1 - rad2) / 2.87
        dt = 0.2
        steps = 10  # 1.6 s / 0.2 = 8
        x, y, th = node.x, node.y, math.radians(node.theta)
        for _ in range(steps):
            x += lin * math.cos(th) * dt
            y += lin * math.sin(th) * dt
            th += ang * dt
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return None
        return Node(
            x,
            y,
            (math.degrees(th) + 180) % 360 - 180,
            rpm=(move[0], move[1]),
            parent=node,
        )

    # line collision‑check reused from your file
    def is_line_collision_free(self, p1, p2, step=2):
        dist = self._dist(p1, p2)
        for i in range(int(dist / step) + 1):
            u = i / max(1, int(dist / step))
            x = p1.x + (p2.x - p1.x) * u
            y = p1.y + (p2.y - p1.y) * u
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return False
        return True

    # ───────────────────────────── main loop ────────────────────────────
    # ───────────────────────────── main loop ────────────────────────────
    def plan(self, width, height, vis=None):
        """Return list[(x,y)] on success, else None."""
        for _ in tqdm(range(self.max_iter), desc="RRT* Planning", unit="nodes"):

            # ------------------------------------------------ sample
            rnd = self.sample_free(width, height)
            near0 = self.nearest(self.tree, rnd)  # KD‑tree query

            # ------------------------------------------------ extend
            best_new, best_dist = None, float("inf")
            for m in self.move_set():
                cand = self.simulate_motion(near0, m)
                if cand is None:
                    continue
                d = self._dist(cand, rnd)
                if d < best_dist:
                    best_new, best_dist = cand, d
            if best_new is None:
                continue

            # ------------------------------------------------ choose parent
            n_pts = len(self.tree) + 1
            r = min(self.gamma_rrt * (math.log(n_pts) / n_pts) ** (1 / self.dim), 40.0)
            near_nodes = self.near(best_new, r)
            best_parent = near0
            min_cost = near0.cost + self._dist(near0, best_new)
            for nbr in near_nodes:
                c = nbr.cost + self._dist(nbr, best_new)
                if c < min_cost and self.is_line_collision_free(nbr, best_new):
                    best_parent, min_cost = nbr, c
            best_new.parent = best_parent
            best_new.cost = min_cost

            # ------------------------------------------------ insert node
            self.tree.append(best_new)
            self._coords.append((best_new.x, best_new.y))
            self._kd = cKDTree(self._coords)  # ← rebuild EVERY insert
            if vis:
                vis.record_edge(best_parent.position(), best_new.position())

            # ------------------------------------------------ rewire
            for nbr in near_nodes:
                new_cost = best_new.cost + self._dist(best_new, nbr)
                if new_cost < nbr.cost and self.is_line_collision_free(best_new, nbr):
                    nbr.parent = best_new
                    nbr.cost = new_cost
                    if vis:
                        vis.record_edge(best_new.position(), nbr.position())

            # ------------------------------------------------ goal check
            if self._dist(best_new, self.goal) < self.threshold:
                self.goal.parent = best_new
                self.goal.cost = best_new.cost + self._dist(best_new, self.goal)
                self.path = self._extract_path()
                if vis:
                    vis.record_path(self.path)
                return self.path

        print("No path found.")
        return None

    # ──────────────────────────── utilities ────────────────────────────
    def _extract_path(self):
        node, out = self.goal, []
        while node is not None:
            out.append((node.x, node.y))
            node = node.parent
        return out[::-1]


# ────────────────────────────────────────────────────────────────────
#  OpenCV‑based visualiser
# ────────────────────────────────────────────────────────────────────
class Visualizer:
    def __init__(
        self,
        map_img: np.ndarray,
        start,
        goal,
        scale: float = 1.0,
        window_name: str = "RRT* explore",
        video_out: str | None = None,
        fps: int = 30,
    ):
        """
        Parameters
        ----------
        map_img : np.ndarray
            BGR image returned by env.create_canvas().
        start, goal : (x, y)
        video_out : str | None
            If not None, path where an MP4 will be written.
        """
        self.base = cv.resize(
            map_img, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST
        ).copy()
        self.canvas = self.base.copy()
        self.start = (int(start[0] * scale), int(start[1] * scale))
        self.goal = (int(goal[0] * scale), int(goal[1] * scale))
        self.window = window_name
        self.scale = scale

        cv.circle(self.canvas, self.start, 4, (0, 255, 0), -1)
        cv.circle(self.canvas, self.goal, 4, (0, 0, 255), -1)

        # ---- video writer --------------------------------------------
        self.writer = None
        if video_out:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            h, w = self.canvas.shape[:2]
            self.writer = cv.VideoWriter(video_out, fourcc, fps, (w, h))

        cv.namedWindow(self.window, cv.WINDOW_NORMAL)
        cv.imshow(self.window, self.canvas)
        cv.waitKey(1)

    # -----------------------------------------------------------------
    def record_edge(self, parent_xy, child_xy, colour=(255, 200, 0)):
        """
        Draw a small line segment as soon as it is generated.
        """
        p = tuple(int(c * self.scale) for c in parent_xy)
        c = tuple(int(c * self.scale) for c in child_xy)
        cv.line(self.canvas, p, c, colour, 1, cv.LINE_AA)
        cv.imshow(self.window, self.canvas)
        if self.writer:
            self.writer.write(self.canvas)
        cv.waitKey(1)  # ~1 ms, keeps window responsive

    # -----------------------------------------------------------------
    def record_path(self, path_xy, colour=(0, 0, 255), thickness=2):
        """
        Draw the final path.  `path_xy` is a list of (x, y) tuples.
        """
        pts = [tuple(int(p * self.scale) for p in xy) for xy in path_xy]
        for i in range(len(pts) - 1):
            cv.line(self.canvas, pts[i], pts[i + 1], colour, thickness, cv.LINE_AA)
        cv.imshow(self.window, self.canvas)
        if self.writer:
            self.writer.write(self.canvas)
        cv.waitKey(0)  # wait for key press

        # Safely close writer / window
        if self.writer:
            self.writer.release()
        try:
            cv.destroyWindow(self.window)
        except cv.error:
            pass  # gracefully handle if already closed


def smooth_path(path, env, step_size=2):
    if not path:
        return []
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if is_line_collision_free(path[i], path[j], env, step_size):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def is_line_collision_free(p1, p2, env, step_size=2):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    steps = max(1, int(dist / step_size))
    for i in range(steps + 1):
        x = x1 + (x2 - x1) * i / steps
        y = y1 + (y2 - y1) * i / steps
        if env.is_in_obstacle(x, y, env.inflated_obs):
            return False
    return True


def draw_raw_and_smoothed_path(map_img, start, goal, raw_path, smoothed_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv.cvtColor(map_img, cv.COLOR_BGR2RGB), origin="lower")
    ax.scatter([start[0]], [start[1]], c="green", s=100)
    ax.scatter([goal[0]], [goal[1]], c="red", s=100)
    if raw_path:
        x, y = zip(*[(pt[0], pt[1]) for pt in raw_path])
        ax.plot(x, y, "gray", linestyle="--", linewidth=2, label="Raw Path")
    if smoothed_path:
        x, y = zip(*[(pt[0], pt[1]) for pt in smoothed_path])
        ax.plot(x, y, "r-", linewidth=3, label="Smoothed Path")
    plt.title("Raw vs Smoothed RRT Path")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_planner(env, start=(0, 150, 0), goal=(540, 150, 0), rpm1=60, rpm2=60):
    canvas = env.create_canvas(540, 300)
    planner = RRTPlanner(rpm1, rpm2, start, goal, env)
    raw_path = planner.plan(canvas.shape[1], canvas.shape[0], vis=None)

    if raw_path:
        formatted_path = []
        for i in range(len(raw_path) - 1):
            x1, y1 = raw_path[i]
            x2, y2 = raw_path[i + 1]
            theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
            formatted_path.append((x1, y1, theta, rpm1, rpm2))
        x_end, y_end = raw_path[-1]
        theta_end = formatted_path[-1][2] if formatted_path else 0
        formatted_path.append((x_end, y_end, theta_end, rpm1, rpm2))
        return formatted_path
    return None


if __name__ == "__main__":
    height, width = 300, 540
    scale_factor = 1.0

    input_handler = UserInput()
    clearance = 0.05
    robot_radius = 2.2

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    rpm1, rpm2 = 60, 60
    start, goal = input_handler.get_start_goal(env, width, height)

    print(f"\nUsing RPMs: ({rpm1}, {rpm2})")
    print(f"Start: {start}, Goal: {goal}")

    vis = Visualizer(canvas, start, goal, scale=1.0, video_out=None)
    planner = RRTPlanner(rpm1, rpm2, start, goal, env)

    print("Starting RRT planning...")
    start_time = time.time()
    raw_path = planner.plan(width, height, vis)
    end_time = time.time()

    if raw_path:
        print("\nGoal reached!")
        print(f"Total path steps: {len(raw_path)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        for pt in raw_path:
            print(f"→ {pt}")
        path_length = compute_path_length(raw_path)
        path_jerk = compute_path_jerkiness(raw_path)
        print(f"Path length   : {path_length:.2f}")
        print(f"Smoothness    : {path_jerk:.4f} radians")

        smoothed_path = smooth_path(raw_path, env)
        if smoothed_path:
            print("\nSmoothed Path:")
            for pt in smoothed_path:
                print(f"→ {pt}")
        else:
            print("\n Could not smooth the path.")
        vis.record_path([(pt[0], pt[1]) for pt in smoothed_path])
        vis.animate_explore()
        vis.animate_path()
        vis.draw_path()
        draw_raw_and_smoothed_path(canvas, start, goal, raw_path, smoothed_path)
    else:
        print("\n No path found.")
