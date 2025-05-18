# Comparative_Analysis_PathPlanning-ENPM661-Final-Project
 
The study benchmarks seven planners (A*, Weighted A*, PRM + Dijkstra, PRM + Weighted A*, RRT, RRT + A*, RRT*) on 2-D obstacle maps while respecting differential-drive kinematics. This section explains **how the repository mirrors – and extends – the methodology in the paper and summarises the final insights.**

## Features
* Runs **seven planners** (RRT, RRT\*, PRM\_Dijkstra, PRM\_Weighted_A\*, RRT\_A\*, A\*, Weighted\_A\*) across **eight obstacle maps** with one command.  
* Caches previous results and only re-runs jobs whose planner source has changed.  
* Computes per-run metrics (success, runtime, path length, jerkiness, safety, …).   
* Parallel evaluation with Python’s `asyncio` + `ProcessPoolExecutor` for near-linear speed-ups on multi-core CPUs.

---

## Directory layout
```

.
├── planners/               # your planner implementations (rrt.py, astar.py, …)
├── map/                    # eight MapEnv subclasses with obstacle layouts
├── utils.py                # metric helpers
├── evaluate\_consistency.py # full benchmark runner
├── validate\_all\_runners.py # quick smoke-tests
├── plots.py                # turns eval\_cache.json → figures
└── README.md               # this file

````

---

## Installation
```bash
git clone https://github.com/kiki101robo/Comparative_Analysis_PathPlanning-ENPM661-Final-Project.git
cd Comparative_Analysis_PathPlanning-ENPM661-Final-Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

---

## Quick start

```bash
# 1 — run the full evaluation (maps × planners)
python evaluate_consistency.py

# 2 — generate charts (To get the visualization of the eval_cache.jason's data)
python plots.py --save png

# 3 — smoke-test only
python validate_all_runners.py
```

---

## Key scripts

### `evaluate_consistency.py`

1. Imports eight map modules and a configurable planner list.
2. Converts physical clearance to pixels and initialises each `MapEnv`.
3. Spawns a **separate process per (map, planner)** job for true parallelism.
4. Records success flag, runtime, path length and jerkiness for every run.
5. Caches successful results together with the planner file’s mtime.
6. Skips unchanged planners on subsequent runs; writes `eval_cache.json`.
7. Prints both per-map comparisons and cross-map summaries to the console.

### `validate_all_runners.py`

1. Loads a single reference map and hard-coded start/goal pose.
2. Iterates through each planner in `PLANNERS`.
3. Dynamically imports `planners.<name>` and calls its `run_planner()`.
4. Asserts the return type to catch API mismatches.
5. Reports / per planner together with any traceback.

### `plots.py`

1. Reads `eval_cache.json` and derives aggregated statistics.
2. Builds seven figures (bar charts, scatter, line, …).
3. Helper `grouped_bars()` keeps multi-planner plots tidy.
4. `--save <fmt>` writes images; default pops up an interactive window.
5. Gracefully errors if the cache is empty or missing.

### `utils.py`

1. `compute_path_length()` – Euclidean sum over way-points.
2. `compute_path_jerkiness()` – mean absolute heading change.
3. `compute_path_safety()` – minimum signed distance to obstacles.
4. Accepts plain `(x, y)` tuples **or** objects with `.x` / `.y` attributes.

### `userinput.py`

1. CLI helper that **validates human inputs** for start/goal, RPMs and clearance.
2. Rejects coordinates outside bounds or inside inflated obstacles.
3. Converts clearance (m) → pixels at 5 mm / px.
4. Re-prompts until data is valid.
5. Re-usable in other interactive demos.

---

## Visualising results

```bash
python plots.py --save pdf   # pdf/png/svg supported
```

Charts (`figure_1.pdf`, …) appear in the working directory.

---

## Inspiration & Project Conclusion

**Research springboard** – The benchmark design is heavily influenced by a recent *Robotic* study that compared PRM, RRT and Voronoi-Diagram planners across 100 procedurally-generated maps, scoring them on path length, computation time, safety and consistency.

### What we borrowed

* Re-created their four core metrics in `utils.py`.
* Tested the same baseline planners (RRT, RRT\*, PRM\_Dijkstra, PRM\_Weighted_A\*, RRT\_A\*, A\*, Weighted\_A\*).
* Mirrored their success criterion (≤ 1.5 × optimal length and collision-free).
* Used map densities in the 50 – 80 % obstacle-occupancy band.
* Averaged results over 30 + random seeds to tame sampling variance.

### Where we extended

* Added **jerkiness** as a fifth metric to capture path smoothness — critical for real robot actuation.
* Replaced MATLAB scripts with a fully-parallel Python pipeline and automatic result caching, cutting wall-time from hours to minutes on an 8-core machine.
* Swapped purely synthetic C-space images for **eight hand-drawn maps** containing cul-de-sacs and bottlenecks that break visibility-graph assumptions.
