import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import random
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
        self.inflated_obs = None

    def obstacles_list(self):
        obstacles = [
            # First vertical block with slit between 125–175
            np.array([[100, 0], [100, 125], [110, 125], [110, 0]]),
            np.array([[100, 175], [100, 300], [110, 300], [110, 175]]),

            # Second vertical block with slit between 100–150
            np.array([[210, 0], [210, 100], [220, 100], [220, 0]]),
            np.array([[210, 150], [210, 250], [220, 250], [220, 150]]),

            # Third vertical block with slit between 180–220
            np.array([[320, 50], [320, 180], [330, 180], [330, 50]]),
            np.array([[320, 220], [320, 300], [330, 300], [330, 220]]),

            # Fourth vertical block with slit between 75–125
            np.array([[430, 0], [430, 75], [440, 75], [440, 0]]),
            np.array([[430, 125], [430, 250], [440, 250], [440, 125]]),

            # L-shaped block 1 (bottom-left)
            np.array([[50, 50], [50, 70], [100, 70], [100, 60], [60, 60], [60, 50]]),

            # L-shaped block 3 (center-right)
            np.array([[400, 140], [400, 160], [440, 160], [440, 150], [410, 150], [410, 140]])
        ]
        return obstacles



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

    def create_canvas(self, width, height):
        planning_canvas = np.full((int(height), int(width), 3), COLOR_BACKGROUND, dtype=np.uint8)
        self.inflated_obs = self.inflate_obstacles()
        self.add_clearance(planning_canvas, self.inflated_obs)
        for obs in self.obstacles:
            pts = np.array(obs, dtype=np.int32).reshape((-1, 1, 2))
            cv.fillPoly(planning_canvas, [pts], COLOR_OBSTACLE)
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
        return False


if __name__ == "__main__":
    height = 300
    width = 540
    clearance = 15  # pixels
    robot_radius = 5  # pixels

    env = MapEnv(height, width, clearance, robot_radius)
    canvas = env.create_canvas(width, height)
    rgb_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_canvas, origin="lower")
    plt.title("Test Map - Obstacle + Clearance")
    plt.axis("on")
    plt.grid(True)
    plt.show()