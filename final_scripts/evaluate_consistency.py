# evaluate_all.py
import asyncio
import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import traceback
from pathlib import Path


import numpy as np

# ---------------------------------------------------------------------------
#  Paths & dynamic imports
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
CACHE_FILE = Path(BASE_DIR) / "eval_cache.json"

from map import mapenv0, mapenv1, mapenv2, mapenv3, mapenv4, mapenv5, mapenv6, mapenv7
from utils import compute_path_length, compute_path_jerkiness

MAP_MODULES = [
    mapenv0,
    mapenv1,
    mapenv2,
    mapenv3,
    mapenv4,
    mapenv5,
    mapenv6,
    mapenv7,
]

PLANNERS = [
    "rrt",
    "rrtStar",
    "rrt_astar",
    "astar",
    "astarweighted",
    "prm_djikstra",
    "prm_weightedAstar",
]

# ---------------------------------------------------------------------------
#  Global configuration
# ---------------------------------------------------------------------------
START = (0, 150, 0)
GOAL = (540, 150, 0)
CLEARANCE_METERS = 0.05
PIXELS_PER_METER = 100
CLEARANCE_PIXELS = int(CLEARANCE_METERS * PIXELS_PER_METER)
ROBOT_RADIUS = 2.2
RPM1 = RPM2 = 60


# ---------------------------------------------------------------------------
#  Worker executed in separate processes
# ---------------------------------------------------------------------------
def _evaluate_single(map_idx: int, planner_name: str):
    """Run one planner on one map and return metrics."""
    print(f"[START] Map {map_idx + 1} – {planner_name}", flush=True)  # ➊

    map_mod = MAP_MODULES[map_idx]
    env = map_mod.MapEnv(
        height=300,
        width=540,
        clearance=CLEARANCE_PIXELS,
        robot_radius=ROBOT_RADIUS,
    )
    env.create_canvas(540, 300)

    planner_mod = importlib.import_module(f"planners.{planner_name}")
    planner_func = getattr(planner_mod, "run_planner")

    t0 = time.time()
    path = planner_func(env, START, GOAL, RPM1, RPM2)
    duration = time.time() - t0

    success = path is not None
    length = compute_path_length(path) if success else 0.0
    jerk = compute_path_jerkiness(path) if success else 0.0

    print(
        f"[DONE ] Map {map_idx + 1} – {planner_name} "
        f"({ 'ok' if success else 'fail' }) {duration:.2f}s",
        flush=True,
    )  # ➋
    sys.stdout.flush()
    return {
        "map_idx": map_idx,
        "planner": planner_name,
        "planner_mtime": Path(planner_mod.__file__).stat().st_mtime,  # NEW
        "success": success,
        "time": duration,
        "length": length,
        "jerk": jerk,
    }


def _planner_mtime(planner_name: str) -> float:
    """Last‑modified time of the planner’s .py file."""
    mod = importlib.import_module(f"planners.{planner_name}")
    return Path(mod.__file__).stat().st_mtime


def load_cache():
    if CACHE_FILE.exists():
        with CACHE_FILE.open("r") as f:
            return json.load(f)
    return []


def save_cache(cache_list):
    tmp = CACHE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cache_list, f, indent=2)
    tmp.replace(CACHE_FILE)


# ---------------------------------------------------------------------------
#  Async orchestration
# ---------------------------------------------------------------------------
async def run_all_evaluations():
    """Run (map, planner) jobs, stream progress, handle errors, update cache."""
    cache = load_cache()
    cached = {
        (c["map_idx"], c["planner"]): c
        for c in cache
        if c["success"] and c["planner_mtime"] == _planner_mtime(c["planner"])
    }

    results, tasks = [], []
    loop = asyncio.get_running_loop()
    total = len(MAP_MODULES) * len(PLANNERS)

    with ProcessPoolExecutor() as pool:
        for m in range(len(MAP_MODULES)):
            for p in PLANNERS:
                key = (m, p)
                if key in cached:
                    results.append(cached[key])
                    print(f"[SKIP ] Map {m+1} – {p}  (cached)", flush=True)
                    continue
                fn = partial(_evaluate_single, m, p)
                tasks.append(loop.run_in_executor(pool, fn))

        remaining = len(tasks)
        print(f"\n>>> {remaining} of {total} jobs need to run <<<\n", flush=True)

        # ---- gather them *as they finish* ----
        for done_idx, fut in enumerate(asyncio.as_completed(tasks), 1):
            try:
                res = await fut
            except Exception:
                print(
                    "[ERROR] A worker crashed:\n" f"{traceback.format_exc()}",
                    flush=True,
                )
                continue

            res["planner_mtime"] = _planner_mtime(res["planner"])
            results.append(res)
            print(
                f"[COLLECT] {done_idx}/{remaining} complete  "
                f"(Map {res['map_idx']+1} – {res['planner']})",
                flush=True,
            )

    # ---- save merged cache atomically ----
    merged = {(r["map_idx"], r["planner"]): r for r in results}
    save_cache(list(merged.values()))
    return results


# ---------------------------------------------------------------------------
#  Reporting helpers
# ---------------------------------------------------------------------------
def print_per_map_comparison(results):
    print("\nPer‑map comparison (relative length: best = 1.00)\n")
    for m_idx in range(len(MAP_MODULES)):
        subset = [r for r in results if r["map_idx"] == m_idx]
        best_len = min(r["length"] for r in subset if r["success"]) or 1.0
        print(f"Map {m_idx + 1}")
        print(
            f"{'Planner':<20} {'Status':<8} {'t (s)':<8} "
            f"{'length':<10} {'jerk':<10} {'rel_len':<8}"
        )
        print("-" * 70)
        for r in subset:
            rel = (r["length"] / best_len) if r["success"] else np.inf
            status = "ok" if r["success"] else "fail"
            print(
                f"{r['planner']:<20} {status:<8} {r['time']:<8.2f} "
                f"{r['length']:<10.2f} {r['jerk']:<10.4f} {rel:<8.2f}"
            )
        print()


def print_cross_map_summary(results):
    print("\nCross‑map summary")
    header = ("Planner", "Succ %", "avg t (s)", "avg rel_len", "avg rel_jerk")
    print(
        f"{header[0]:<20} {header[1]:<8} {header[2]:<10} {header[3]:<12} {header[4]:<12}"
    )
    print("-" * 70)

    for planner in PLANNERS:
        subset = [r for r in results if r["planner"] == planner]
        total = len(subset)
        succ = [r for r in subset if r["success"]]

        if total == 0:
            continue  # Skip planners with no data at all

        succ_rate = 100 * len(succ) / total
        avg_time = np.mean([r["time"] for r in subset])

        rel_lens, rel_jerks = [], []

        for m_idx in range(len(MAP_MODULES)):
            map_runs = [r for r in results if r["map_idx"] == m_idx and r["success"]]
            if not map_runs:
                continue

            best_len = min(r["length"] for r in map_runs)
            best_jerk = min(r["jerk"] for r in map_runs) or 1.0

            this_run = next((r for r in map_runs if r["planner"] == planner), None)
            if this_run is None:
                continue  # this planner did not succeed on this map

            rel_lens.append(this_run["length"] / best_len)
            rel_jerks.append(this_run["jerk"] / best_jerk)

        avg_rel_len = np.mean(rel_lens) if rel_lens else float("inf")
        avg_rel_jerk = np.mean(rel_jerks) if rel_jerks else float("inf")

        print(
            f"{planner:<20} {succ_rate:<8.1f} {avg_time:<10.2f} "
            f"{avg_rel_len:<12.2f} {avg_rel_jerk:<12.2f}"
        )


# ---------------------------------------------------------------------------
#  Entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = asyncio.run(run_all_evaluations())
    print_per_map_comparison(all_results)
    print_cross_map_summary(all_results)
