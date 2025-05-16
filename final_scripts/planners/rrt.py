import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import math
import heapq as hp
import time
from map.mapenv3 import MapEnv
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


class RRTPlanner:
    def __init__(self, rpm1, rpm2, start, goal, env, max_iter=12000, goal_sample_rate=0.1):
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.env = env
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.tree = [self.start]
        self.edges = []
        self.path = []
        self.threshold = 15.0
        self.kdtree = None

        # ✅ Precomputed constants
        self.rad_per_rpm = 2 * math.pi / 60
        self.wheel_radius = 3.3
        self.L = 2.87

    def build_kdtree(self):
        self.kdtree = KDTree([(n.x, n.y) for n in self.tree])

    def move_set(self):
        return [[0, self.rpm1], [0, self.rpm2], [self.rpm1, 0], [self.rpm2, 0],
                [self.rpm1, self.rpm2], [self.rpm2, self.rpm1],
                [self.rpm1, self.rpm1], [self.rpm2, self.rpm2]]

    def nearest(self, target_node):
        if self.kdtree is None:
            return min(self.tree, key=lambda n: np.hypot(n.x - target_node.x, n.y - target_node.y))
        _, idx = self.kdtree.query((target_node.x, target_node.y))
        return self.tree[idx]

    def sample_free(self, width, height, max_attempts=50):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal

        for _ in range(max_attempts):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            if not self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                theta = np.random.uniform(-180, 180)
                return Node(x, y, theta)

        # ⚠️ Fallback: return goal if unable to find a free spot
        return self.goal

    def simulate_motion(self, node, move):
        rad_rpm1 = self.rad_per_rpm * move[0]
        rad_rpm2 = self.rad_per_rpm * move[1]

        lin_vel = self.wheel_radius * (rad_rpm1 + rad_rpm2) / 2
        ang_vel = self.wheel_radius * (rad_rpm1 - rad_rpm2) / self.L

        dt = 0.2            # ✅ Increased timestep (fewer steps = faster)
        total_time = 2.0    # ✅ Optional: reduced total time
        steps = int(total_time / dt)

        x, y, theta = node.x, node.y, node.theta
        theta_rad = math.radians(theta)

        for _ in range(steps):
            x += lin_vel * math.cos(theta_rad) * dt
            y += lin_vel * math.sin(theta_rad) * dt
            theta_rad += ang_vel * dt
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return None

        theta_new_deg = (math.degrees(theta_rad) + 180) % 360 - 180
        return Node(x, y, theta_new_deg, rpm=(move[0], move[1]), parent=node)

    def plan(self, width, height, vis=None):

        for i in tqdm(range(self.max_iter), desc="RRT Planning", unit="nodes"):
            if self.kdtree is None or len(self.tree) % 100 == 0:
                self.build_kdtree()

            rand_node = self.sample_free(width, height)
            nearest_node = self.nearest(rand_node)

            best_new_node = None
            min_dist = float("inf")

            for move in self.move_set():
                new_node = self.simulate_motion(nearest_node, move)
                if new_node:
                    dx = new_node.x - rand_node.x
                    dy = new_node.y - rand_node.y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < min_dist:
                        best_new_node = new_node
                        min_dist = dist_sq


            if best_new_node:
                self.tree.append(best_new_node)
                self.edges.append((nearest_node, best_new_node))
                if vis:
                    vis.record_edge(nearest_node.position(), best_new_node.position())

                if np.hypot(best_new_node.x - self.goal.x, best_new_node.y - self.goal.y) < self.threshold:
                    self.goal.parent = best_new_node
                    self.tree.append(self.goal)
                    self.edges.append((best_new_node, self.goal))
                    if vis:
                        vis.record_edge(best_new_node.position(), self.goal.position())
                    self.path = self.reconstruct_path()
                    if vis:
                        vis.record_path([(n.x, n.y) for n in self.path])
                    return self.path

        print("No path found.")
        return None

    def reconstruct_path(self):
        path = []
        node = self.goal
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(self.start)
        return path[::-1]

    def is_line_collision_free(self, p1, p2):
        steps = int(np.hypot(p2.x - p1.x, p2.y - p1.y) / 2)
        for i in range(steps + 1):
            x = p1.x + (p2.x - p1.x) * i / steps
            y = p1.y + (p2.y - p1.y) * i / steps
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return False
        return True

    def smooth_path(self, path):
        if not path:
            return []
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.is_line_collision_free(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

class Visualizer:
    def __init__(self, map, start, goal, scale=1.0):
        self.map = map.copy()
        self.start = start
        self.goal = goal
        self.scale = scale
        self.edges = []
        self.path = []

    def record_edge(self, parent, child):
        self.edges.append((parent, child))

    def record_path(self, path):
        self.path = path

    def animate_explore(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal')
        plt.title("RRT Exploration")
        plt.legend()

        def update(i):
            if i < len(self.edges):
                p, c = self.edges[i]
                ax.plot([p[0], c[0]], [p[1], c[1]], 'c-', linewidth=0.5, alpha=0.7)
            return []

        
        anim = FuncAnimation(fig, update, frames=len(self.edges), interval=5)
        plt.show()

    def animate_path(self):
        if not self.path:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal')
        x, y = zip(*[(pt[0], pt[1]) for pt in self.path])
        ax.plot(x, y, 'r-', linewidth=3, label='Final Path')
        point, = ax.plot([], [], 'bo', ms=10)

        def update(frame):
            point.set_data([x[frame]], [y[frame]])
            return point,

        # ✅ Assign animation to variable
        anim = FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)
        plt.title("RRT Path Animation")
        plt.legend()
        plt.show()

    def draw_path(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal')
        if self.path:
            x, y = zip(*[(pt[0], pt[1]) for pt in self.path])
            ax.plot(x, y, 'r-', linewidth=3, label='Final Path')
        plt.title("Final RRT Path")
        plt.grid(True)
        plt.legend()
        plt.show()


def run_planner(env, start=(0, 150, 0), goal=(540, 150, 0), rpm1=60, rpm2=60):
    vis = None
    planner = RRTPlanner(rpm1, rpm2, start, goal, env)
    raw_path = planner.plan(env.canvas[1], env.canvas[0], vis)
    if raw_path:
        return [(node.x, node.y, node.theta, node.rpm[0], node.rpm[1]) for node in raw_path]
    return None


if __name__ == "__main__":
    height = 300
    width = 540
    scale_factor = 1.0

    input_handler = UserInput()
    clearance = 0.05
    robot_radius = 2.2

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    rpm1, rpm2 = 60,60
    start, goal = input_handler.get_start_goal(env, width, height)

    print(f"\nUsing RPMs: ({rpm1}, {rpm2})")
    print(f"Start: {start}, Goal: {goal}")

    vis = Visualizer(canvas, start, goal, scale=scale_factor)
    planner = RRTPlanner(rpm1, rpm2, start, goal, env)

    print("Starting RRT planning...")
    start_time = time.time()
    path = planner.plan(width, height, vis)
    end_time = time.time()

    if path:
        print("\nGoal reached!")
        print(f"Total path steps: {len(path)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")
        print("Final path:")
        for pt in path:
            print(f"→ ({pt.x:.2f}, {pt.y:.2f}, {pt.theta:.2f}, {pt.rpm[0]}, {pt.rpm[1]})")

        path_length = compute_path_length(path)
        path_jerk = compute_path_jerkiness(path)
        print(f"{'Smoothness':<10}: {path_jerk:.4f} radians")

        vis.record_path([pt.position() for pt in path])
        vis.animate_explore()
        vis.animate_path()
        vis.draw_path()
    else:
        print("\nNo path found.")
