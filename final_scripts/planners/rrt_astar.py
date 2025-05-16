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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Updated based on your folder structure
from map.mapenv0 import MapEnv
from userinput import UserInput
from utils import compute_path_length, compute_path_jerkiness
from scipy.spatial import KDTree




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
    def __init__(self, rpm1, rpm2, start, goal, env, max_iter=20000, goal_sample_rate=0.1):
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.start = Node(*start)
        self.start.cost = 0
        self.goal = Node(*goal)
        self.env = env
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.tree = [self.start]
        self.edges = []
        self.path = []
        self.threshold = 15.0
        self.graph = {}

        # KDTree-related attributes
        self.kdtree_points = [self.start.position()]
        self.kdtree = KDTree(self.kdtree_points)
        self.kdtree_update_freq = 20
        self.kdtree_insert_count = 0

    def move_set(self):
        return [[0, self.rpm1], [0, self.rpm2], [self.rpm1, 0], [self.rpm2, 0],
                [self.rpm1, self.rpm2], [self.rpm2, self.rpm1],
                [self.rpm1, self.rpm1], [self.rpm2, self.rpm2]]

    def nearest(self, nodes, target_node):
        dist, idx = self.kdtree.query([target_node.x, target_node.y])
        return self.tree[idx]

    def sample_free(self, width, height):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal
        while True:
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            if not self.env.is_in_obstacle(x, y, self.env.inflated_obs):
                theta = np.random.uniform(-180, 180)
                return Node(x, y, theta)

    def simulate_motion(self, node, move):
        rad_rpm1 = (2 * math.pi / 60) * move[0]
        rad_rpm2 = (2 * math.pi / 60) * move[1]
        lin_vel = 3.3 * (rad_rpm1 + rad_rpm2) / 2
        ang_vel = 3.3 * (rad_rpm1 - rad_rpm2) / 2.87
        dt = 0.1
        total_time = 2.0
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
        for _ in tqdm(range(self.max_iter), desc="RRT Planning", unit="nodes"):
            rand_node = self.sample_free(width, height)
            nearest_node = self.nearest(self.tree, rand_node)

            best_new_node = None
            min_dist = float("inf")

            for move in self.move_set():
                new_node = self.simulate_motion(nearest_node, move)
                if new_node:
                    dist = np.hypot(new_node.x - rand_node.x, new_node.y - rand_node.y)
                    if dist < min_dist:
                        best_new_node = new_node
                        min_dist = dist

            if best_new_node:
                best_new_node.cost = nearest_node.cost + min_dist
                self.tree.append(best_new_node)

                # Update KDTree data
                self.kdtree_points.append(best_new_node.position())
                self.kdtree_insert_count += 1
                if self.kdtree_insert_count >= self.kdtree_update_freq:
                    self.kdtree = KDTree(self.kdtree_points)
                    self.kdtree_insert_count = 0

                self.edges.append((nearest_node, best_new_node))
                if vis:
                    vis.record_edge(nearest_node.position(), best_new_node.position())

                # Build graph for A*
                u, v = nearest_node.position(), best_new_node.position()
                self.graph.setdefault(u, []).append((v, min_dist))
                self.graph.setdefault(v, []).append((u, min_dist))  # bidirectional


                if np.hypot(best_new_node.x - self.goal.x, best_new_node.y - self.goal.y) < self.threshold:
                    goal_node = self.goal
                    goal_node.parent = best_new_node
                    self.tree.append(goal_node)
                    self.edges.append((best_new_node, goal_node))
                    if vis:
                        vis.record_edge(best_new_node.position(), goal_node.position())

                    u, v = best_new_node.position(), goal_node.position()
                    dist = np.hypot(goal_node.x - best_new_node.x, goal_node.y - best_new_node.y)
                    self.graph.setdefault(u, []).append((v, dist))
                    self.graph.setdefault(v, []).append((u, dist))

                    # --- NEW: Compare raw RRT vs A* extracted path ---
                    rrt_path = self.backtrack_path(goal_node)
                    astar_path = self.astar_extract_path()

                    if vis:
                        vis.record_path(astar_path)

                    print("\n--- PATH COMPARISON ---")
                    print(f"RRT Raw Path Length   : {len(rrt_path)}")
                    print(f"A* Extracted Path Len : {len(astar_path)}")
                    print(f"Raw Path Distance     : {compute_path_length(rrt_path):.2f} px")
                    print(f"A* Path Distance      : {compute_path_length(astar_path):.2f} px")
                    print("------------------------\n")

                    self.path = astar_path
                    return astar_path

    def backtrack_path(self, node):
        path = []
        while node:
            path.append(node.position())
            node = node.parent
        path.reverse()
        return path

    def astar_extract_path(self):
        start = self.start.position()
        goal = self.goal.position()
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = hp.heappop(open_set)

            if np.hypot(current[0] - goal[0], current[1] - goal[1]) < self.threshold:
                goal = current
                break

            for neighbor, cost in self.graph.get(current, []):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                    hp.heappush(open_set, (f_score, neighbor))

        # Reconstruct path
        path = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

class Visualizer:
    """
    Visualizer for RRT: Handles drawing the exploration tree and the final path.
    """

    def __init__(self, map, start, goal, scale=1.0):
        self.map = map.copy()
        self.start = start
        self.goal = goal
        self.scale = scale
        self.edges = []  # List of (parent, child) edges
        self.path = []   # Final path

    def record_edge(self, parent, child):
        """
        Record an edge for exploration animation.
        """
        self.edges.append((parent, child))

    def record_path(self, path):
        """
        Record the final path (list of points).
        """
        self.path = path

    def animate_explore(self):
        """
        Animate the RRT exploration with persistent tree growth (branching visible).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start', marker='o')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal', marker='o')
        plt.title("RRT Exploration Animation (with Branching)")
        plt.legend()

        def update(i):
            if i < len(self.edges):
                parent, child = self.edges[i]
                ax.plot([parent[0], child[0]], [parent[1], child[1]], 'c-', linewidth=0.5, alpha=0.7)
            return []

        ani = FuncAnimation(fig, update, frames=len(self.edges), interval=5, blit=False)
        plt.show(block=True)
        plt.close()

    def animate_path(self):
        """
        Animate the final path with a moving robot dot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start', marker='o')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal', marker='o')

        if not self.path:
            return

        path_x, path_y = zip(*[(pt[0], pt[1]) for pt in self.path])
        ax.plot(path_x, path_y, 'r-', linewidth=3, label='Final Path')
        plt.title("RRT Path Animation")
        plt.legend()

        point, = ax.plot([], [], 'bo', ms=10)

        def update(frame):
            point.set_data([path_x[frame]], [path_y[frame]])
            return point,

        ani = FuncAnimation(fig, update, frames=len(path_x), interval=50, blit=True)
        plt.show(block=True)
        plt.close()

    def draw_path(self):
        """
        Draws the static final path.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        rgb_map = cv.cvtColor(self.map, cv.COLOR_BGR2RGB)
        ax.imshow(rgb_map, origin="lower")
        ax.scatter([self.start[0]], [self.start[1]], c='green', s=100, label='Start', marker='o')
        ax.scatter([self.goal[0]], [self.goal[1]], c='red', s=100, label='Goal', marker='o')
        if self.path:
            path_x, path_y = zip(*[(pt[0], pt[1]) for pt in self.path])
            ax.plot(path_x, path_y, 'r-', linewidth=3, label='Final Path')
        plt.title("Final RRT Path (Static)")
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
        plt.close()

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
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    steps = max(1, int(dist / step_size))
    for i in range(steps + 1):
        x = x1 + (x2 - x1) * i / steps
        y = y1 + (y2 - y1) * i / steps
        if env.is_in_obstacle(x, y, env.inflated_obs):
            return False
    return True
def draw_raw_and_smoothed_path(map_img, start, goal, raw_path, smoothed_path):
    """
    Plots both the raw and smoothed RRT paths on the same map.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    rgb_map = cv.cvtColor(map_img, cv.COLOR_BGR2RGB)
    ax.imshow(rgb_map, origin="lower")
    ax.scatter([start[0]], [start[1]], c='green', s=100, label='Start', marker='o')
    ax.scatter([goal[0]], [goal[1]], c='red', s=100, label='Goal', marker='o')

    if raw_path:
        raw_x, raw_y = zip(*[(pt[0], pt[1]) for pt in raw_path])
        ax.plot(raw_x, raw_y, 'gray', linestyle='--', linewidth=2, label='Raw Path')

    if smoothed_path:
        smooth_x, smooth_y = zip(*[(pt[0], pt[1]) for pt in smoothed_path])
        ax.plot(smooth_x, smooth_y, 'r-', linewidth=3, label='Smoothed Path')

    plt.title("Raw vs Smoothed RRT Path")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import time

    height, width = 300, 540
    scale_factor = 1.0

    input_handler = UserInput()
    clearance = input_handler.get_clearance()
    robot_radius = 2.2

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)

    # Show original map after clearance
    rgb_map = cv.cvtColor(canvas.copy(), cv.COLOR_BGR2RGB)
    plt.ion()
    plt.imshow(rgb_map, origin="lower")
    plt.title("Original Map with Obstacles")
    plt.show()
    plt.show(block=True)
    plt.close()

    rpm1, rpm2 = input_handler.get_rpm()
    start, goal = input_handler.get_start_goal(env, width, height)

    print(f"\nUsing RPMs: ({rpm1}, {rpm2})")
    print(f"Start: {start}, Goal: {goal}")

    # Show map with start and goal points
    rgb_with_points = cv.cvtColor(canvas.copy(), cv.COLOR_BGR2RGB)
    plt.ion()
    plt.imshow(rgb_with_points, origin="lower")
    plt.scatter([start[0]], [start[1]], c='green', s=100, label='Start', marker='o')
    plt.scatter([goal[0]], [goal[1]], c='red', s=100, label='Goal', marker='o')
    plt.title("Start and Goal Positions")
    plt.legend()
    plt.show()
    plt.show(block=True)
    plt.close()

    vis = Visualizer(canvas, start, goal, scale=scale_factor)
    planner = RRTPlanner(rpm1, rpm2, start, goal, env)

    print("Starting RRT planning...")
    start_time = time.time()
    raw_path = planner.plan(width, height, vis)
    end_time = time.time()

    if raw_path:
        print("\nGoal reached!")
        print(f"Total raw path steps: {len(raw_path)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")
        print("Raw Path:")
        for pt in raw_path:
            print(f"→ {pt}")

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
    else:
        print("\n No path found.")
        smoothed_path = []

    draw_raw_and_smoothed_path(canvas, start, goal, raw_path, smoothed_path)



# if __name__ == "__main__":
    

#     height, width = 300, 540
#     scale_factor = 1.0

#     input_handler = UserInput()
#     clearance = input_handler.get_clearance()
#     robot_radius = 2.2

#     env = MapEnv(height, width, clearance, robot_radius)
#     canvas = env.create_canvas(width, height)

#     # Show original map
#     rgb_map = cv.cvtColor(canvas.copy(), cv.COLOR_BGR2RGB)
#     plt.ion()
#     plt.imshow(rgb_map, origin="lower")
#     plt.title("Original Map with Obstacles")
#     plt.show()
#     plt.show(block=True)
#     plt.close()

#     rpm1, rpm2 = input_handler.get_rpm()
#     start, goal = input_handler.get_start_goal(env, width, height)

#     print(f"\nUsing RPMs: ({rpm1}, {rpm2})")
#     print(f"Start: {start}, Goal: {goal}")

#     # Show map with start and goal
#     rgb_with_points = cv.cvtColor(canvas.copy(), cv.COLOR_BGR2RGB)
#     plt.ion()
#     plt.imshow(rgb_with_points, origin="lower")
#     plt.scatter([start[0]], [start[1]], c='green', s=100, label='Start', marker='o')
#     plt.scatter([goal[0]], [goal[1]], c='red', s=100, label='Goal', marker='o')
#     plt.title("Start and Goal Positions")
#     plt.legend()
#     plt.show()
#     plt.show(block=True)
#     plt.close()

#     vis = Visualizer(canvas, start, goal, scale=scale_factor)
#     planner = RRTPlanner(rpm1, rpm2, start, goal, env)

#     print("Starting RRT planning...")
#     start_time = time.time()
#     raw_path = planner.plan(width, height, vis)
#     end_time = time.time()

#     if raw_path:
#         print("\nGoal reached!")
#         print(f"Total path steps: {len(raw_path)}")
#         print(f"Time taken: {end_time - start_time:.2f} seconds\n")
#         print("Final path:")
#         for pt in raw_path:
#             print(f"→ {pt}")
#         path_length = compute_path_length(raw_path)
#         path_jerk = compute_path_jerkiness(raw_path)
#         print(f"{'Smoothness':<10}: {path_jerk:.4f} radians")
#         vis.record_path([(pt[0], pt[1]) for pt in raw_path])
#         vis.animate_explore()
#         vis.animate_path()
#         vis.draw_path()
#     else:
#         print("\nNo path found.")

#     # Draw both raw and empty smoothed path
#     draw_raw_and_smoothed_path(canvas, start, goal, raw_path, smoothed_path=[])
#     # # Compute path safety
#     # binary_obstacle_map = env.get_binary_obstacle_map()
#     # safety_score = compute_path_safety(raw_path, binary_obstacle_map)
#     # print(f"Path safety (average distance to obstacles): {safety_score:.2f} pixels")
# def run_planner(env, start=(0, 150, 0), goal=(540, 150, 0), rpm1=60, rpm2=60):
#     """
#     Interface function for consistency evaluation script.
#     Returns:
#         List of (x, y, theta, rpm1, rpm2) tuples using raw RRT path
#     """
#     canvas = env.create_canvas(540, 300)
#     planner = RRTPlanner(rpm1, rpm2, start, goal, env)
#     raw_path = planner.plan(canvas.shape[1], canvas.shape[0], vis=None)

#     if raw_path:
#         formatted_path = []
#         for i in range(len(raw_path) - 1):
#             x1, y1 = raw_path[i]
#             x2, y2 = raw_path[i + 1]
#             theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
#             formatted_path.append((x1, y1, theta, rpm1, rpm2))
#         # Add the final point with the last known heading
#         x_end, y_end = raw_path[-1]
#         theta_end = formatted_path[-1][2] if formatted_path else 0
#         formatted_path.append((x_end, y_end, theta_end, rpm1, rpm2))
#         return formatted_path

#     return None
