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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from map.mapenv0 import MapEnv
from userinput import UserInput
from utils import compute_path_length, compute_path_jerkiness

class PRMPlanner:
    def __init__(self, map_env, canvas, start, goal, rpm1, rpm2, num_samples=500, k=10):
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

        from joblib import Parallel, delayed
        def validate_edge(i, j):
            if self.is_motion_collision_free(self.nodes[i], self.nodes[j]):
                dist = np.linalg.norm(np.array(self.nodes[i]) - np.array(self.nodes[j]))
                return i, j, dist
            return None

        results = Parallel(n_jobs=-1)(delayed(validate_edge)(i, j) for i, j in pairs_to_check)
        for result in results:
            if result:
                i, j, dist = result
                self.edges[i].append((j, dist))
                self.edges[j].append((i, dist))

    def get_actions(self):
        return [[0, self.rpm1], [0, self.rpm2], [self.rpm1, 0], [self.rpm2, 0],
                [self.rpm1, self.rpm2], [self.rpm2, self.rpm1],
                [self.rpm1, self.rpm1], [self.rpm2, self.rpm2]]

    def is_motion_collision_free(self, p1, p2, steps=10):
        for i in range(steps + 1):
            u = i / steps
            x = p1[0] * (1 - u) + p2[0] * u
            y = p1[1] * (1 - u) + p2[1] * u
            if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                return False
        return True

    def weighted_astar(self, epsilon=1.0):
        start_idx = len(self.nodes) - 2
        goal_idx = len(self.nodes) - 1
        dist = {i: float('inf') for i in range(len(self.nodes))}
        dist[start_idx] = 0
        prev = {i: None for i in range(len(self.nodes))}

        def heuristic(i):
            x1, y1 = self.nodes[i]
            x2, y2 = self.nodes[goal_idx]
            return np.linalg.norm([x1 - x2, y1 - y2])

        visited = set()
        pq = [(epsilon * heuristic(start_idx), start_idx)]

        while pq:
            _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal_idx:
                break
            for v, cost in self.edges[u]:
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev[v] = u
                    f = dist[v] + epsilon * heuristic(v)
                    heapq.heappush(pq, (f, v))

        path = []
        current = goal_idx
        while current is not None:
            path.append(self.nodes[current])
            current = prev[current]
        path.reverse()
        return path if path and path[0] == (self.start[0], self.start[1]) else None

    def simulate_dd_path(self, path):
        sim_path = []
        for i in range(len(path) - 1):
            x, y = path[i]
            x_goal, y_goal = path[i+1]
            best_result = None
            min_dist = float('inf')
            for move in self.get_actions():
                x_new, y_new, theta_new = self.motion_sim((x, y, 0), move)
                dist = np.linalg.norm([x_new - x_goal, y_new - y_goal])
                if dist < min_dist and self.is_motion_collision_free((x, y), (x_new, y_new)):
                    min_dist = dist
                    best_result = (x_new, y_new, theta_new, move[0], move[1])
            if best_result:
                sim_path.append(best_result)
        return sim_path
    def visualize_roadmap(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.canvas.copy(), cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.set_title("PRM Roadmap Visualization")

        # Draw nodes
        xs, ys = zip(*self.nodes)
        ax.scatter(xs, ys, s=5, c='cyan', alpha=0.6, label="Sampled Nodes")

        # Draw edges
        for i in self.edges:
            for j, _ in self.edges[i]:
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)

        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start', marker='o')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal', marker='o')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()



class Visualizer:
    def __init__(self, map, start, goal):
        self.map = map.copy()
        self.start = start
        self.goal = goal
        self.explored_nodes = []
        self.path = []

    def record_exploration(self, node):
        self.explored_nodes.append(node)

    def record_path(self, path):
        self.path = path

    def animate_explore(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100)
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100)
        xs, ys = zip(*self.explored_nodes)
        ax.scatter(xs, ys, c='blue', s=5, alpha=0.4)
        plt.title("PRM Exploration")
        plt.show()

    def animate_path(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100)
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100)
        x, y = zip(*[(pt[0], pt[1]) for pt in self.path])
        ax.plot(x, y, 'r-', linewidth=3)
        point, = ax.plot([], [], 'bo', ms=10)

        def update(frame):
            point.set_data([x[frame]], [y[frame]])
            return point,

        FuncAnimation(fig, update, frames=len(x), interval=50, blit=True)
        plt.title("PRM Path Animation")
        plt.show()

    def draw_path(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100)
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100)
        if self.path:
            x, y = zip(*[(pt[0], pt[1]) for pt in self.path])
            ax.plot(x, y, 'r-', linewidth=3)
        plt.title("Final PRM Path")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    height, width = 300, 540
    input_handler = UserInput()
    clearance = input_handler.get_clearance()
    robot_radius = 2.2
    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    rpm1, rpm2 = input_handler.get_rpm()
    start, goal = input_handler.get_start_goal(env, width, height)
    epsilon = input_handler.get_epsilon()

    print(f"\nUsing RPMs: ({rpm1}, {rpm2}), Epsilon: {epsilon}")
    print(f"Start: {start}, Goal: {goal}")

    planner = PRMPlanner(env, canvas, start, goal, rpm1, rpm2, num_samples=1500, k=10)
    vis = Visualizer(canvas, start, goal)

    start_time = time.time()
    planner.build_roadmap()
    planner.visualize_roadmap()

    for u in planner.edges:
        for v, _ in planner.edges[u]:
            vis.record_exploration(planner.nodes[u])
            vis.record_exploration(planner.nodes[v])

    vis.animate_explore()

    path = planner.weighted_astar(epsilon=epsilon)
    end_time = time.time()

    if path:
        sim_path = planner.simulate_dd_path(path)
        if sim_path:
            print("\nGoal reached!")
            print(f"Total path steps: {len(sim_path)}")
            print(f"Total planning time: {end_time - start_time:.2f} seconds\n")
            for pt in sim_path:
                print(f"→ {pt}")
            path_length = compute_path_length(path)
            path_jerk = compute_path_jerkiness(path)
            print(f"Smoothness : {path_jerk:.4f} radians")

            vis.record_path(sim_path)
            vis.animate_path()
            vis.draw_path()
        else:
            print("\nCould not simulate DD path from roadmap.")
    else:
        print("\nNo valid path found using Weighted A*.")


def run_planner(env, start=(0, 150, 0), goal=(540, 150, 0), rpm1=60, rpm2=60):
    canvas = env.create_canvas(540, 300)
    planner = PRMPlanner(env, canvas, start, goal, rpm1, rpm2, num_samples=1500, k=10)
    planner.build_roadmap()
    path = planner.weighted_astar(epsilon=1.5)
    if path:
        return planner.simulate_dd_path(path)
    return None
