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
            np.array([[50, 200], [50, 260], [110, 260], [110, 200]]),    # Wide block
            np.array([[180, 70], [180, 120], [200, 120], [200, 70]]),    # Small vertical
            np.array([[250, 250], [250, 280], [320, 280], [320, 250]]),  # Long horizontal
            np.array([[350, 40], [350, 130], [370, 130], [370, 40]]),    # Mid vertical
            np.array([[420, 160], [420, 200], [500, 200], [500, 160]]),  # Large block

            # New additions: thinner and longer in X-direction
            np.array([[135, 205], [135, 215], [185, 215], [185, 205]]),  # Long thin filler near center
            np.array([[295, 185], [295, 195], [340, 195], [340, 185]]),  # Long thin horizontal between blocks
            np.array([[215, 35], [215, 45], [265, 45], [265, 35]]),      # Top horizontal bar left
            np.array([[395, 225], [395, 235], [455, 235], [455, 225]]),  # Bottom-right extended horizontal
            np.array([[125, 145], [125, 155], [175, 155], [175, 145]])   # Mid-left horizontal filler
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