#!/usr/bin/env python3
"""
Batch evaluation script for COMPASS.

Runs eval_driver.py for each combination of:
  - run_name (model)
  - target_size (fixed sweep list, default [6])
  - num_agents: 2, 3, 5
  - decentralized_sensing: False / True
  - no_collision: True / False

After all runs complete, reads the per-run CSV files and prints a
summary table for: avgunc, stdunc, minnvisit, avgnvisit, stdnvisit, stdgapvisit
"""

import os
import sys
import csv
import subprocess
import itertools
import numpy as np

# ──────────────────────────────────────────────────────────────
# Sweep configuration  –  edit here
# ──────────────────────────────────────────────────────────────
SWEEP = {
    "run_name":              ["decentralized_nocollision_hgt003", "centralized_nocollision_hgt003"],
    "target_size":           [6],
    "num_agents":            [2, 3, 5],
    "decentralized_sensing": [False, True],
    "no_collision":          [True, False],
}

# Fixed eval settings (passed through env vars; override as needed)
FIXED = {
    "NUM_EVAL":     "30",
    "NUM_META":     "16",
    "BUDGET_SIZE":  "30",
    "GRAPH_SIZE":   "200",
    "HISTORY_SIZE": "100",
    "TARGET_SPEED": str(1 / 20.0),
    "SAVE_IMG_GAP": "0",   # disable image saving for speed
}

# Metrics to display in the summary table
DISPLAY_METRICS = ["avgunc", "stdunc", "minnvisit", "avgnvisit", "stdnvisit", "avggapvisit", "stdgapvisit"]

# Where this script lives (COMPASS project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT  = os.path.join(PROJECT_ROOT, "evals", "eval_driver.py")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def result_dir_for(run_name, decentral, num_agents, no_collision):
    """Absolute result directory for a run (unique per num_agents AND no_collision so metric CSVs do not clash).

    eval_driver.py appends to the metric CSV, so every sweep dimension must be
    part of the directory name; sharing a directory across no_collision settings
    pools their episodes into one file and corrupts the summary means.
    """
    suffix = "_decentral" if decentral else ""
    nc_suffix = "_nc1" if no_collision else "_nc0"
    return os.path.join(PROJECT_ROOT, f"{run_name}{suffix}_na{num_agents}{nc_suffix}")


def result_csv_path(run_name, target_size, decentral, num_agents, no_collision, budget=30, graph=200, history=100, speed=20):
    """Return the expected CSV path that eval_driver.py writes."""
    return os.path.join(
        result_dir_for(run_name, decentral, num_agents, no_collision), "metric",
        f"tgt{target_size}_speed{speed}_b{budget}_g{graph}_h{history}_metric.csv"
    )


def read_csv_means(csv_path, metrics):
    """Read a metric CSV and return {metric: mean_value} for the requested metrics."""
    if not os.path.exists(csv_path):
        return {m: float("nan") for m in metrics}

    columns = {m: [] for m in metrics}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for m in metrics:
                try:
                    columns[m].append(float(row[m]))
                except (KeyError, ValueError):
                    pass

    return {m: float(np.mean(v)) if v else float("nan") for m, v in columns.items()}


def run_eval(run_name, target_size, decentral, no_collision, num_agents):
    """
    Launch eval_driver.py as a subprocess with settings injected via
    environment variables.  eval_driver.py reads these via COMPASS_EVAL_*
    env vars (see patch below in eval_driver.py / arguments.py).
    """
    result_dir = result_dir_for(run_name, decentral, num_agents, no_collision)

    env = os.environ.copy()
    env.update({
        "COMPASS_EVAL_RUN_NAME":      run_name,
        "COMPASS_EVAL_TARGET_SIZE":   str(target_size),
        "COMPASS_EVAL_NUM_AGENTS":    str(num_agents),
        "COMPASS_EVAL_DECENTRALIZED": "1" if decentral else "0",
        "COMPASS_EVAL_RESULT_PATH":   result_dir,
        "COMPASS_EVAL_NO_COLLISION":  "1" if no_collision else "0",
        **{f"COMPASS_EVAL_{k}": v for k, v in FIXED.items()},
    })

    cmd   = [sys.executable, EVAL_SCRIPT]
    label = (f"model={run_name}  tgt={target_size}  na={num_agents}  "
             f"decentral={decentral}  nc={no_collision}")
    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"{'='*70}")

    ret = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print(f"  [WARNING] eval_driver.py exited with code {ret.returncode} for {label}")
    return ret.returncode


def print_table(results):
    """Print a nicely formatted summary table."""
    col_w_label = 72
    col_w_val   = 12

    header = f"{'Config':<{col_w_label}}" + "".join(f"{m:>{col_w_val}}" for m in DISPLAY_METRICS)
    sep    = "-" * len(header)

    print(f"\n\n{'='*len(header)}")
    print("  SUMMARY TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for (run_name, tgt, decentral, nc, na), vals in sorted(results.items()):
        label = (f"{run_name} / tgt={tgt} / na={na} / "
                 f"{'decentral' if decentral else 'central'} / "
                 f"nc={nc}")
        row = f"{label:<{col_w_label}}"
        for m in DISPLAY_METRICS:
            v = vals.get(m, float("nan"))
            row += f"{v:>{col_w_val}.4f}"
        print(row)

    print(sep)
    print()


# ──────────────────────────────────────────────────────────────
# Env-var patching for eval_driver / arguments
#
# eval_driver.py imports arguments.py at module level, so we can't
# easily monkeypatch.  Instead, we inject a small shim at the top of
# the subprocess that overrides arg_eval attributes after import.
# We do this by prepending a sitecustomize-style override via PYTHONPATH.
# ──────────────────────────────────────────────────────────────

SHIM_DIR  = os.path.join(PROJECT_ROOT, "_eval_shim")
SHIM_FILE = os.path.join(SHIM_DIR, "sitecustomize.py")

SHIM_TEMPLATE = """\
# Auto-generated shim – do not edit manually.
import os as _os
import sys as _sys

# Add project root (cwd) to sys.path so 'arguments' is importable at startup time.
# eval_driver.py sets cwd=PROJECT_ROOT when launching the subprocess.
_project_root = _os.getcwd()
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

def _patch():
    try:
        import arguments as _args
        a = _args.arg_eval

        v = _os.environ.get("COMPASS_EVAL_RUN_NAME")
        if v:
            a.run_name   = v
            a.model_path = f"models/{v}"
            # result_path may be overridden below
            a.result_path = v

        v = _os.environ.get("COMPASS_EVAL_RESULT_PATH")
        if v:
            a.result_path = v

        v = _os.environ.get("COMPASS_EVAL_TARGET_SIZE")
        if v:
            a.target_size = int(v)

        v = _os.environ.get("COMPASS_EVAL_DECENTRALIZED")
        if v is not None:
            a.decentralized_sensing = (v == "1")
            _args.arg.decentralized_sensing = (v == "1")

        v = _os.environ.get("COMPASS_EVAL_NUM_EVAL")
        if v:
            a.num_eval = int(v)

        v = _os.environ.get("COMPASS_EVAL_NUM_META")
        if v:
            a.num_meta = int(v)

        v = _os.environ.get("COMPASS_EVAL_BUDGET_SIZE")
        if v:
            a.budget_size = int(v)

        v = _os.environ.get("COMPASS_EVAL_GRAPH_SIZE")
        if v:
            a.graph_size = int(v)

        v = _os.environ.get("COMPASS_EVAL_HISTORY_SIZE")
        if v:
            a.history_size = int(v)

        v = _os.environ.get("COMPASS_EVAL_TARGET_SPEED")
        if v:
            a.target_speed = float(v)

        v = _os.environ.get("COMPASS_EVAL_SAVE_IMG_GAP")
        if v:
            a.save_img_gap = int(v)

        v = _os.environ.get("COMPASS_EVAL_NO_COLLISION")
        if v:
            a.no_collision = (v == "1")

        v = _os.environ.get("COMPASS_EVAL_NUM_AGENTS")
        if v:
            na = int(v)
            a.num_agents = na
            _args.arg.num_agents = na

        v = _os.environ.get("COMPASS_EVAL_CI_OMEGA")
        if v:
            # float weight, or the string 'trace' for per-exchange line search
            a.ci_fusion_omega = v if v == "trace" else float(v)
            _args.arg.ci_fusion_omega = a.ci_fusion_omega

        v = _os.environ.get("COMPASS_EVAL_COMM_INTERVAL")
        if v:
            a.comm_exchange_interval = int(v)
            _args.arg.comm_exchange_interval = int(v)

        v = _os.environ.get("COMPASS_EVAL_ALPHA_OVERRIDE")
        if v:
            a.temporal_alpha_override = float(v)
            _args.arg.temporal_alpha_override = float(v)

        v = _os.environ.get("COMPASS_EVAL_MAPPER_IMPL")
        if v:
            a.mapper_impl = v
            _args.arg.mapper_impl = v

    except Exception as _e:
        print(f"[shim] Warning: could not patch arg_eval: {_e}")

_patch()
"""


def setup_shim():
    os.makedirs(SHIM_DIR, exist_ok=True)
    with open(SHIM_FILE, "w") as f:
        f.write(SHIM_TEMPLATE)


def teardown_shim():
    if os.path.exists(SHIM_FILE):
        os.remove(SHIM_FILE)
    if os.path.isdir(SHIM_DIR):
        try:
            os.rmdir(SHIM_DIR)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    combinations = list(itertools.product(
        SWEEP["run_name"],
        SWEEP["target_size"],
        SWEEP["num_agents"],
        SWEEP["decentralized_sensing"],
        SWEEP["no_collision"],
    ))

    print(f"Batch evaluation: {len(combinations)} configurations")
    for i, (m, t, na, d, nc) in enumerate(combinations, 1):
        print(f"  {i:2d}. model={m}  tgt={t}  na={na}  decentral={d}  nc={nc}")

    setup_shim()

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = SHIM_DIR + (os.pathsep + old_pythonpath if old_pythonpath else "")

    results = {}
    failed  = []

    try:
        for run_name, target_size, num_agents, decentral, no_collision in combinations:
            key = (run_name, target_size, decentral, no_collision, num_agents)
            rc  = run_eval(run_name, target_size, decentral, no_collision, num_agents)

            csv_path = result_csv_path(run_name, target_size, decentral, num_agents, no_collision)
            results[key] = read_csv_means(csv_path, DISPLAY_METRICS)

            if rc != 0 or any(np.isnan(v) for v in results[key].values()):
                failed.append(key)
    finally:
        teardown_shim()
        if old_pythonpath:
            os.environ["PYTHONPATH"] = old_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)

    print_table(results)

    summary_path = os.path.join(PROJECT_ROOT, "batch_eval_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "target_size", "decentralized", "no_collision", "num_agents"] + DISPLAY_METRICS
        )
        for (run_name, tgt, decentral, nc, na), vals in sorted(results.items()):
            row = [run_name, tgt, int(decentral), int(nc), na] + [
                vals.get(m, float("nan")) for m in DISPLAY_METRICS
            ]
            writer.writerow(row)
    print(f"Summary saved to: {summary_path}")

    if failed:
        print(f"\n[WARNING] {len(failed)} configuration(s) had errors or missing data:")
        for k in failed:
            print(f"  model={k[0]}  tgt={k[1]}  decentral={k[2]}  nc={k[3]}  na={k[4]}")


if __name__ == "__main__":
    main()
