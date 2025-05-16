import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

# Color constants
COLOR_OBSTACLE = (0, 0, 0)
COLOR_CLEARANCE = (128, 128, 128)
COLOR_BACKGROUND = (255, 255, 255)


class MapEnv:
    def __init__(self, height, width, clearance, robot_radius):
        self.canvas = (height, width)
        self.clearance = clearance
        self.robot_radius = robot_radius
        self.obstacles = self.obstacles_list()
        self.circle_obstacles = self.circle_list()
        self.inflated_obs = None

    def obstacles_list(self):
        obstacles = [
            np.array([[80, 80], [80, 150], [120, 150], [120, 80]]),
            np.array([[200, 100], [200, 180], [215, 180], [215, 100]]),
            np.array([[250, 50], [250, 90], [300, 90], [300, 50]]),
            np.array([[350, 120], [350, 250], [360, 250], [360, 120]]),
            np.array([[450, 60], [450, 100], [490, 100], [490, 60]])
        ]
        return obstacles

    def circle_list(self):
        return [
            (150, 220, 10),
            (400, 40, 8),
            (470, 220, 6)
        ]

    def inflate_obstacles(self):
        margin = self.clearance + self.robot_radius + 2
        inflated_obs = []
        for obstacle in self.obstacles:
            inflated = []
            n = len(obstacle)
            for i in range(n):
                x, y = obstacle[i]
                prev_x, prev_y = obstacle[(i - 1) % n]
                next_x, next_y = obstacle[(i + 1) % n]
                dx1, dy1 = x - prev_x, y - prev_y
                nx1, ny1 = -dy1, dx1
                L1 = np.hypot(nx1, ny1)
                nx1, ny1 = (nx1 / L1, ny1 / L1) if L1 > 0 else (0, 0)
                dx2, dy2 = next_x - x, next_y - y
                nx2, ny2 = -dy2, dx2
                L2 = np.hypot(nx2, ny2)
                nx2, ny2 = (nx2 / L2, ny2 / L2) if L2 > 0 else (0, 0)
                nx, ny = (nx1 + nx2), (ny1 + ny2)
                L = np.hypot(nx, ny)
                nx, ny = (nx / L, ny / L) if L > 0 else (0, 0)
                inflated.append([x + margin * nx, y + margin * ny])
            if inflated:
                inflated_obs.append(np.array(inflated))
        return inflated_obs

    def add_clearance(self, planning_canvas, inflated_obs):
        for obs in inflated_obs:
            pts = np.array(obs, dtype=np.int32).reshape((-1, 1, 2))
            cv.fillPoly(planning_canvas, [pts], COLOR_CLEARANCE)
        margin = self.clearance + self.robot_radius + 2
        for cx, cy, r in self.circle_obstacles:
            cv.circle(planning_canvas, (cx, cy), int(r + margin), COLOR_CLEARANCE, -1)

    def create_canvas(self, width, height):
        planning_canvas = np.full((int(height), int(width), 3), COLOR_BACKGROUND, dtype=np.uint8)
        self.inflated_obs = self.inflate_obstacles()
        self.add_clearance(planning_canvas, self.inflated_obs)
        for obs in self.obstacles:
            pts = np.array(obs, dtype=np.int32).reshape((-1, 1, 2))
            cv.fillPoly(planning_canvas, [pts], COLOR_OBSTACLE)
        for cx, cy, r in self.circle_obstacles:
            cv.circle(planning_canvas, (cx, cy), int(r), COLOR_OBSTACLE, -1)
        return planning_canvas

    def is_in_obstacle(self, x, y, inflated_obs):
        margin = self.clearance + self.robot_radius
        for obs in inflated_obs:
            inside = False
            n = len(obs)
            j = n - 1
            for i in range(n):
                xi, yi = obs[i]
                xj, yj = obs[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            if inside:
                return True
            for xi, yi in obs:
                if math.hypot(x - xi, y - yi) < margin:
                    return True
        for cx, cy, r in self.circle_obstacles:
            if math.hypot(x - cx, y - cy) < (r + margin):
                return True
        return False


if __name__ == "__main__":
    height = 300
    width = 540
    clearance = 15
    robot_radius = 5

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)
    rgb_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_canvas, origin="lower")
    plt.title("Test Map - Rectangles + Inflated Circles + Clearance")
    plt.axis("on")
    plt.grid(True)
    plt.show()
