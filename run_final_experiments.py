#!/usr/bin/env python3
"""
Camera-ready (final round) eval-only experiments for RA-L 26-3428. All NC-on,
N=6 targets, speed 20, budget 30, graph 200, 32 episodes (seeds 1..32).
Reuses the env-var shim from run_batch_eval.py; every config gets its own
final_* result dir (the metric CSV writer appends, so dirs are never shared).

  T1  PROPOSED-Decentralized R in {2,3,5} (omega=0.5, exchange every 3 steps):
      reference-belief avgunc + robots' own-belief avgunc_agents
  T2  accuracy/calibration diagnostics (KFGP_EVAL_EXTRA_METRICS=1, written to
      *_metric_ext.csv) for PROPOSED-Cent/Dec, AUCTION on the grid belief,
      the full-covariance mapper (Cent, R=3) and COVERAGE (T3)
  T3  COVERAGE (belief-free) on the grid belief, graph seeded per episode
      (KFGP_EVAL_GRAPH_SEED=1) and episodes logged for the cross-codebase
      identity check against COMPASS (experiments/run_final_coverage.py there)
  T4  sensing/communication counters (same ext CSV; decentralized rows)

Usage:
  conda run -n compass_env python run_final_experiments.py            # run all
  conda run -n compass_env python run_final_experiments.py summary    # tables
  conda run -n compass_env python run_final_experiments.py check-replay
"""

import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

import run_batch_eval as rb

PROJECT_ROOT = rb.PROJECT_ROOT
EVAL_SCRIPT = rb.EVAL_SCRIPT
COMPASS_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "COMPASS_ORIG", "COMPASS_orig")

CENT = "centralized_nocollision_hgt003"
DEC = "decentralized_nocollision_hgt003"

# (tag, run_name, decentral, num_agents, extra_env)
CONFIGS = []
for na in [2, 3, 5]:
    CONFIGS.append((f"T1_dec_na{na}", DEC, True, na, {}))
for na in [2, 3, 5]:
    CONFIGS.append((f"T2_cent_na{na}", CENT, False, na, {}))
for na in [2, 3, 5]:
    CONFIGS.append((f"T2_auction_grid_na{na}", CENT, False, na, {"KFGP_EVAL_METHOD": "AUCTION"}))
CONFIGS.append(("T2_fullcov_cent_na3", CENT, False, 3, {"COMPASS_EVAL_MAPPER_IMPL": "full"}))
# T3 v1 dirs (final_T3_coverage_grid_na{N}) are SUPERSEDED: they were run with
# the graph seeded by the episode seed itself, which made graph nodes / start
# positions coincide with the target waypoints (correlated draws). v2 seeds the
# graph from an offset stream (eval_worker.py, KFGP_EVAL_GRAPH_SEED).
for na in [2, 3, 5]:
    CONFIGS.append((f"T3v2_coverage_grid_na{na}", CENT, False, na,
                    {"KFGP_EVAL_METHOD": "COVERAGE", "KFGP_EVAL_GRAPH_SEED": "1",
                     "KFGP_EVAL_EPISODE_LOG": os.path.join(PROJECT_ROOT, f"final_T3v2_coverage_grid_na{na}", "episodes")}))


def result_dir(tag):
    return os.path.join(PROJECT_ROOT, f"final_{tag}")


def metric_csv(tag, ext=True):
    name = "tgt6_speed20_b30_g200_h100_metric" + ("_ext" if ext else "") + ".csv"
    return os.path.join(result_dir(tag), "metric", name)


def run_config(tag, run_name, decentral, na, extra):
    if os.path.exists(metric_csv(tag, ext=False)):
        print(f"skip {tag}: metric CSV exists")
        return 0
    env = os.environ.copy()
    env.update({
        "COMPASS_EVAL_RUN_NAME": run_name,
        "COMPASS_EVAL_TARGET_SIZE": "6",
        "COMPASS_EVAL_NUM_AGENTS": str(na),
        "COMPASS_EVAL_DECENTRALIZED": "1" if decentral else "0",
        "COMPASS_EVAL_NO_COLLISION": "1",
        "COMPASS_EVAL_RESULT_PATH": result_dir(tag),
        "KFGP_EVAL_EXTRA_METRICS": "1",
        **{f"COMPASS_EVAL_{k}": v for k, v in rb.FIXED.items()},
        **extra,
    })
    print(f"\n===== {tag}: model={run_name} na={na} decentral={decentral} {extra} =====", flush=True)
    return subprocess.run([sys.executable, EVAL_SCRIPT], env=env, cwd=PROJECT_ROOT).returncode


def read_rows(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def stats(rows, key):
    v = np.array([float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")])
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    ci = 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(np.mean(v)), ci, len(v)


SUMMARY_KEYS = ["avgunc", "avgunc_agents", "minnvisit", "avgnvisit",
                "rmse_all_ref", "abserr_tgt_ref", "nees_tgt_ref", "nees_all_ref", "cov2_tgt_ref", "cov2_all_ref",
                "rmse_all_agents", "abserr_tgt_agents", "nees_tgt_agents", "nees_all_agents",
                "cov2_tgt_agents", "cov2_all_agents",
                "meas_per_robot_step", "cells_per_robot_step", "detections_per_robot_step",
                "frac_meas_neg_dropped", "neg_dropped_per_drop_event",
                "comm_exchanges", "comm_bytes_per_exchange", "packet_bytes", "packet_itemsize"]


def summary():
    print("===== final_* summary (mean ± 95% CI) =====")
    for tag, *_ in CONFIGS:
        rows = read_rows(metric_csv(tag))
        if rows is None:
            print(f"{tag}: MISSING")
            continue
        print(f"\n{tag}  (n={len(rows)})")
        for k in SUMMARY_KEYS:
            m, ci, n = stats(rows, k)
            if n:
                print(f"   {k:28s} {m:10.4f} ± {ci:.4f}")


def _load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


# Visit counts under COMPASS's detection rule (40x40 cells, target in a cell
# whose centre lies in the footprint polygon, first target per cell), evaluated
# with COMPASS's own footprint code on logged poses/targets. Run inside the
# COMPASS tree. Only cells that contain a target are tested, in COMPASS's
# (i, j) loop order, which reproduces its loop exactly.
_COMPASS_RULE = r'''
import glob, json, os, sys
import numpy as np
from env import Env
from arguments import arg
E = object.__new__(Env)
cs = 1.0 / 40
def observed(x, y, h, tg):
    verts = E._get_fov_polygon_vertices(np.array([x, y]), h, arg.camera_pitch, arg.camera_hfov,
                                        arg.camera_vfov, arg.flight_altitude)
    cand = set()
    for tx, ty in tg:
        for i in {int(np.floor(tx / cs)), int(np.ceil(tx / cs)) - 1}:
            for j in {int(np.floor(ty / cs)), int(np.ceil(ty / cs)) - 1}:
                if 0 <= i < 40 and 0 <= j < 40:
                    cand.add((i, j))
    o = np.zeros(len(tg), dtype=bool)
    for i, j in sorted(cand):
        if not E._point_in_polygon(np.array([(i + 0.5) * cs, (j + 0.5) * cs]), verts):
            continue
        for t, (tx, ty) in enumerate(tg):
            if i * cs <= tx <= (i + 1) * cs and j * cs <= ty <= (j + 1) * cs:
                o[t] = True
                break
    return o
out = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "ep*.npz"))):
    ml = np.load(p)["meas_log"]
    nt = (ml.shape[1] - 8) // 3
    v = np.zeros(nt, dtype=int)
    for row in ml:
        v += observed(row[2], row[3], row[4], row[8 + nt:8 + 3 * nt].reshape(nt, 2))
    out[os.path.basename(p)] = v.tolist()
print(json.dumps(out))
'''


def compass_rule_visits(log_dir):
    res = subprocess.run([sys.executable, "-c", _COMPASS_RULE, log_dir], cwd=COMPASS_ROOT,
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def check_replay(atol=0.0):
    """Compare, episode by episode, the KFGP COVERAGE logs with the COMPASS
    COVERAGE logs: graph, routing adjacency, routes, target trajectories,
    measurement poses/target positions (exact equality by default), and visit
    counts natively and under a common (COMPASS) detection rule."""
    for na in [2, 3, 5]:
        kdir = os.path.join(result_dir(f"T3v2_coverage_grid_na{na}"), "episodes")
        cdir = os.path.join(COMPASS_ROOT, f"final_coverage_pertarget_v2_na{na}", "episodes")
        keps = sorted(glob.glob(os.path.join(kdir, "ep*.npz")))
        if not keps:
            print(f"na={na}: no KFGP logs in {kdir}")
            continue
        rule = compass_rule_visits(kdir)
        keys = ["graph+starts", "adjacency", "cycle", "routes", "target_traj", "meas_poses+targets",
                "logged_obs==visits(both)", "visits_native", "visits_common_rule"]
        n_ok = dict.fromkeys(keys, 0)
        d_min, d_avg = [], []
        n_ep = 0
        for kp in keps:
            cp = os.path.join(cdir, os.path.basename(kp))
            if not os.path.exists(cp):
                print(f"na={na} {os.path.basename(kp)}: missing COMPASS log")
                continue
            k, c = _load(kp), _load(cp)
            n_ep += 1

            def same(x, y):
                return x.shape == y.shape and np.allclose(x, y, atol=atol, rtol=0)

            km, cm = k["meas_log"], c["meas_log"]
            nt = len(k["visit_counts"])
            # meas_log columns: t, agent, x, y, heading, 3 counters, observed[nt], targets[2*nt]
            pose_cols = list(range(5)) + list(range(8 + nt, 8 + 3 * nt))
            ok = [
                same(k["node_coords"], c["node_coords"]) and same(k["start_positions"], c["start_positions"]),
                same(k["adjacency"], c["adjacency"]),
                same(k["coverage_cycle"], c["coverage_cycle"]),
                same(k["routes"], c["routes"]),
                same(k["target_traj"], c["target_traj"]),
                km.shape == cm.shape and same(km[:, pose_cols], cm[:, pose_cols]),
                np.array_equal(km[:, 8:8 + nt].sum(0), k["visit_counts"])
                and np.array_equal(cm[:, 8:8 + nt].sum(0), c["visit_counts"]),
                np.array_equal(k["visit_counts"], c["visit_counts"]),
                rule[os.path.basename(kp)] == c["visit_counts"].tolist(),
            ]
            for key, o in zip(keys, ok):
                n_ok[key] += int(o)
            d_min.append(int(k["visit_counts"].min()) - int(c["visit_counts"].min()))
            d_avg.append(float(k["visit_counts"].mean() - c["visit_counts"].mean()))
        print(f"na={na}: {n_ep} paired episodes; identical: " + ", ".join(f"{k}={v}/{n_ep}" for k, v in n_ok.items()))
        if n_ep:
            print(f"   native visit counts (KFGP 32x32 rule - COMPASS 40x40 rule): "
                  f"min-visit diff mean {np.mean(d_min):+.3f} (|.| {np.mean(np.abs(d_min)):.3f}), "
                  f"avg-visit diff mean {np.mean(d_avg):+.3f} (|.| {np.mean(np.abs(d_avg)):.3f})")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "check-replay":
        check_replay()
        return
    only = set(sys.argv[1:])
    rb.setup_shim()
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = rb.SHIM_DIR + (os.pathsep + old_pp if old_pp else "")
    failed = []
    try:
        for tag, run_name, decentral, na, extra in CONFIGS:
            if only and tag not in only:
                continue
            if run_config(tag, run_name, decentral, na, extra) != 0:
                failed.append(tag)
    finally:
        rb.teardown_shim()
        if old_pp:
            os.environ["PYTHONPATH"] = old_pp
        else:
            os.environ.pop("PYTHONPATH", None)
    summary()
    if failed:
        print(f"\n[WARNING] failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
