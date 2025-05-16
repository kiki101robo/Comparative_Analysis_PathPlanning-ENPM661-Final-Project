import sys

class UserInput:
    """
    Handles user interaction for collecting start/goal poses, RPMs, and clearance.

    Attributes:
        start (tuple): Start pose (x, y, theta).
        goal (tuple): Goal pose (x, y, theta).
        rpm1 (float): Left wheel RPM.
        rpm2 (float): Right wheel RPM.
        clearance (float): Robot clearance in pixels.
    """

    def __init__(self, start=None, goal=None, rpm1=None, rpm2=None, clearance=None):
        self.start = start
        self.goal = goal
        self.rpm1 = rpm1
        self.rpm2 = rpm2
        self.clearance = clearance

    def get_start_goal(self, env, width, height):
        """
        Prompts the user for start and goal positions, validates against map bounds and obstacles.
        """
        valid = False
        while not valid:
            try:
                start_input = input("Enter Start Coordinates (x,y,theta): ")
                goal_input = input("Enter Goal Coordinates (x,y,theta): ")
                sx, sy, stheta = map(float, start_input.split(","))
                gx, gy, gtheta = map(float, goal_input.split(","))
                if (0 <= sx <= width and 0 <= gx <= width and
                        0 <= sy <= height and 0 <= gy <= height):
                    if env.is_in_obstacle(sx, sy, env.inflated_obs):
                        print(f"Start point {(sx, sy)} is invalid or in an obstacle.")
                    elif env.is_in_obstacle(gx, gy, env.inflated_obs):
                        print(f"Goal point {(gx, gy)} is invalid or in an obstacle.")
                    else:
                        self.start = (sx, sy, stheta)
                        self.goal = (gx, gy, gtheta)
                        valid = True
                else:
                    print("Coordinates out of bounds. Please try again.")
            except Exception as e:
                print(f"Invalid format: {e}. Please use the format: x,y,theta")
        return self.start, self.goal

    def get_rpm(self):
        """
        Prompts and validates user input for RPM values.
        """
        valid = False
        while not valid:
            try:
                rpm_input = input("Enter RPM values in format (rpm1, rpm2): ")
                rpm1, rpm2 = map(float, rpm_input.split(","))
                if 0 <= rpm1 <= 60 and 0 <= rpm2 <= 60:
                    self.rpm1 = rpm1
                    self.rpm2 = rpm2
                    valid = True
                else:
                    print("RPM values out of range, try again")
            except Exception as e:
                print(f"Invalid format: {e}. Please use the format: rpm1,rpm2")
        return self.rpm1, self.rpm2

    def get_clearance(self):
        """
        Prompts user for clearance in meters and converts it to pixels.
        """
        valid = False
        while not valid:
            try:
                clearance = float(input("Enter robot clearance (in meters): "))
                if clearance >= 0:
                    PIXEL_TO_METER = 0.005  # 1 pixel = 5 mm
                    self.clearance = clearance / PIXEL_TO_METER
                    valid = True
                else:
                    print("Clearance must be non-negative.")
            except Exception as e:
                print(f"Invalid input: {e}")
        return self.clearance

    def get_epsilon(self):
        """
        Prompts user for epsilon value used in Weighted A*.
        """
        while True:
            try:
                epsilon = float(input("Enter epsilon value for Weighted A* (>1): "))
                if epsilon >= 1:
                    return epsilon
                else:
                    print("Epsilon must be ≥ 1.")
            except:
                print("Invalid input. Please enter a numeric value.")
