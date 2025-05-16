import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2 as cv
import matplotlib.pyplot as plt
import math
import heapq as hp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from map.mapenv2 import MapEnv
from userinput import UserInput
from utils import compute_path_length, compute_path_jerkiness


class Node:
    def __init__(self, x, y, cost2come, cost2go, theta=0, parent=None):
        self.x = x
        self.y = y
        self.cost2come = cost2come
        self.cost2go = cost2go
        self.theta = theta
        self.parent = parent

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def get_key(self):
        resolution = 6
        return (int(self.x // resolution), int(self.y // resolution), int(self.theta // 30))

    def total_cost(self):
        return self.cost2come + self.cost2go


class aStar:
    def __init__(self, rpm1, rpm2, theta, start, goal, env):
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.theta = theta
        self.start = start
        self.goal = goal
        self.env = env

    def move_set(self):
        return [
            [0, self.rpm1], [0, self.rpm2], [self.rpm1, 0], [self.rpm2, 0],
            [self.rpm1, self.rpm2], [self.rpm2, self.rpm1],
            [self.rpm1, self.rpm1], [self.rpm2, self.rpm2]
        ]

    def heuristic(self, x, y):
        return math.hypot(x - self.goal[0], y - self.goal[1])

    def motion_sim(self, node, move):
        rad_rpm1 = (2 * math.pi / 60) * move[0]
        rad_rpm2 = (2 * math.pi / 60) * move[1]
        lin_vel = 3.3 * (rad_rpm1 + rad_rpm2) / 2
        ang_vel = 3.3 * (rad_rpm1 - rad_rpm2) / 2.87
        dt = 0.8
        x, y, theta = node.x, node.y, math.radians(node.theta)

        x += lin_vel * math.cos(theta) * dt
        y += lin_vel * math.sin(theta) * dt
        theta += ang_vel * dt
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        if math.hypot(x - node.x, y - node.y) < 0.5:
            return None
        if self.env.is_in_obstacle(x, y, self.env.inflated_obs):
            return None

        cost2come = node.cost2come + math.hypot(x - node.x, y - node.y) + abs(ang_vel) * dt * 0.1
        return Node(x, y, cost2come, self.heuristic(x, y), math.degrees(theta), node), move

    def neighbors(self, node):
        return [
            (new_state, move)
            for move in self.move_set()
            if (result := self.motion_sim(node, move)) and 0 <= result[0].x <= 540 and 0 <= result[0].y <= 300
            for new_state, move in [result]
        ]

    def astar_search(self, vis=None):
        open_list, visited = [], {}
        start_node = Node(self.start[0], self.start[1], 0, self.heuristic(self.start[0], self.start[1]), self.start[2])
        hp.heappush(open_list, (start_node.total_cost(), start_node))

        with tqdm(desc="A* Search Progress", unit="node") as pbar:
            while open_list:
                _, current_node = hp.heappop(open_list)
                key = current_node.get_key()
                if key in visited and visited[key].total_cost() <= current_node.total_cost():
                    continue
                visited[key] = current_node
                if vis:
                    vis.record_exploration(current_node)
                if self.heuristic(current_node.x, current_node.y) <= 5.0:
                    path = self.reconstruct_path(current_node)
                    if vis:
                        vis.record_path(path)
                    return path
                for neighbor, move in self.neighbors(current_node):
                    n_key = neighbor.get_key()
                    if n_key not in visited or visited[n_key].total_cost() > neighbor.total_cost():
                        neighbor.move = move
                        hp.heappush(open_list, (neighbor.total_cost(), neighbor))
                pbar.update(1)
        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            move = getattr(node, 'move', [0, 0])
            path.append((node.x, node.y, node.theta, move[0], move[1]))
            node = node.parent
        return path[::-1]


class Visualizer:
    def __init__(self, map, start, goal, scale=1.0):
        self.map = map.copy()
        self.start = start
        self.goal = goal
        self.scale = scale
        self.explored_nodes = []
        self.path = []

    def record_exploration(self, node):
        self.explored_nodes.append((node.x, node.y))

    def record_path(self, path):
        self.path = path

    def animate_explore(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal')
        plt.title("Exploration Animation")
        plt.legend()

        x_batch, y_batch = [], []
        for x, y in self.explored_nodes:
            x_batch.append(x)
            y_batch.append(y)
            if len(x_batch) >= 2000:
                ax.scatter(x_batch, y_batch, c='blue', s=5, alpha=0.5)
                x_batch.clear()
                y_batch.clear()
                plt.draw()
                plt.pause(0.00001)
        if x_batch:
            ax.scatter(x_batch, y_batch, c='blue', s=5, alpha=0.5)
        plt.draw()
        plt.show()

    def animate_path(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100)
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100)
        path_x, path_y = [pt[0] for pt in self.path], [pt[1] for pt in self.path]
        ax.plot(path_x, path_y, c='red', linewidth=3)
        plt.title("Path Animation")
        line, = ax.plot([], [], 'bo', ms=10)

        def update(frame):
            line.set_data([path_x[frame]], [path_y[frame]])
            return line,

        FuncAnimation(fig, update, frames=range(len(self.path)), interval=50, blit=True)
        plt.show()

    def draw_path(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv.cvtColor(self.map, cv.COLOR_BGR2RGB), origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal')
        if self.path:
            path_x, path_y = zip(*[(pt[0], pt[1]) for pt in self.path])
            ax.plot(path_x, path_y, c='red', linewidth=3, label='Path')
        plt.title("Final Path")
        plt.grid(True)
        plt.legend()
        plt.show()


def run_planner(env, start=(0, 150, 0), goal=(540, 150, 0), rpm1=60, rpm2=60):
    vis = None
    planner = aStar(rpm1, rpm2, start[2], start, goal, env)
    return planner.astar_search(vis)


if __name__ == "__main__":
    height, width = 300, 540
    input_handler = UserInput()
    clearance = input_handler.get_clearance()
    robot_radius = 2.2

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    input_handler.get_rpm()
    start, goal = input_handler.get_start_goal(env, width, height)
    rpm1, rpm2 = input_handler.rpm1, input_handler.rpm2

    vis = Visualizer(canvas, start, goal)
    planner = aStar(rpm1, rpm2, start[2], start, goal, env)
    path = planner.astar_search(vis)

    if path:
        print("\nFinal path:")
        for pt in path:
            print(f"→ {pt}")
        print(f"\nTotal path length: {compute_path_length(path):.2f} units")
        print(f"Smoothness: {compute_path_jerkiness(path):.4f} radians")
        vis.animate_explore()
        vis.animate_path()
        vis.draw_path()
    else:
        print("No path found.")
