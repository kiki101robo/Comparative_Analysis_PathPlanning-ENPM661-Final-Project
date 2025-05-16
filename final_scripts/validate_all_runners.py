import importlib
import traceback
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from map import mapenv0
from userinput import UserInput

START = (0, 150, 0)
GOAL = (540, 150, 0)
RPM1, RPM2 = 60, 60
CLEARANCE_PIXELS = int(0.05 * 100)
ROBOT_RADIUS = 2.2

PLANNERS = [
    "rrt",
    "rrt_astar",
    "astar",
    "astarweighted",
    "prm_djikstra",
    "prm_weightedAstar"
]

for planner_name in PLANNERS:
    print(f"\n🔎 Validating planner: {planner_name}")
    try:
        env = mapenv0.MapEnv(height=300, width=540, clearance=CLEARANCE_PIXELS, robot_radius=ROBOT_RADIUS)
        _ = env.create_canvas(540, 300)

        planner_mod = importlib.import_module(f"planners.{planner_name}")
        planner_func = getattr(planner_mod, "run_planner")

        path = planner_func(env, START, GOAL, RPM1, RPM2)
        assert path is None or isinstance(path, list), "run_planner() must return a list or None"

        print(f"✅ {planner_name} passed.")
    except Exception as e:
        print(f"❌ {planner_name} failed with error:\n{traceback.format_exc()}")
