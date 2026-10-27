#!/usr/bin/env python3
"""
Cheap eval-only ablation sweeps for the RA-L revision (all NC-on, matching the
paper's primary configuration). Reuses the env-var shim from run_batch_eval.py.

  A1  CI weight omega       : decentralized R=3, omega in {0.3, 0.7, trace}
  A2  comm interval         : decentralized R=3, interval in {1, 5, 10, 999999}
  A6  process update off    : alpha=1.0, centralized + decentralized R=3
  A3  team-size scaling     : R in {8, 10, 15}, centralized + decentralized

Baseline points (omega=0.5, interval=3, Matern alpha, R in {2,3,5}) already
exist from the main benchmark. Results go to ablation_* result dirs, one per
config, so nothing is ever appended across settings.

Usage: conda run -n compass_env python run_ablation_sweeps.py
"""

import csv
import os
import subprocess
import sys

import numpy as np

import run_batch_eval as rb

PROJECT_ROOT = rb.PROJECT_ROOT
EVAL_SCRIPT  = rb.EVAL_SCRIPT

CENT = "centralized_nocollision_hgt003"
DEC  = "decentralized_nocollision_hgt003"

# (tag, run_name, decentral, num_agents, extra_env)
CONFIGS = []

# A1 — CI omega (fusion only exists in the decentralized deployment)
for omega in ["0.3", "0.7", "trace"]:
    CONFIGS.append((f"A1_omega{omega}", DEC, True, 3,
                    {"COMPASS_EVAL_CI_OMEGA": omega}))

# A2 — communication interval (999999 = never exchange -> fusion off)
for interval in ["1", "5", "10", "999999"]:
    CONFIGS.append((f"A2_comint{interval}", DEC, True, 3,
                    {"COMPASS_EVAL_COMM_INTERVAL": interval}))

# A6 — process update disabled (alpha = 1: no forgetting)
CONFIGS.append(("A6_alpha1_cent", CENT, False, 3, {"COMPASS_EVAL_ALPHA_OVERRIDE": "1.0"}))
CONFIGS.append(("A6_alpha1_dec",  DEC,  True,  3, {"COMPASS_EVAL_ALPHA_OVERRIDE": "1.0"}))

# A3 — zero-shot team-size scaling
for na in [8, 10, 15]:
    CONFIGS.append((f"A3_cent_na{na}", CENT, False, na, {}))
    CONFIGS.append((f"A3_dec_na{na}",  DEC,  True,  na, {}))

# A4 — full-covariance mapper (centralized deployment; the decentralized one
# exchanges diagonal packets by design). Compare against the main NC-on rows.
for na in [2, 3, 5]:
    CONFIGS.append((f"A4_fullcov_cent_na{na}", CENT, False, na,
                    {"COMPASS_EVAL_MAPPER_IMPL": "full"}))

# A1b/A2b — reruns of the fusion ablations WITH the per-agent belief metric
# (avgunc_agents column): the reference avgunc is computed on a centralized
# evaluation mapper and cannot see local belief quality, which is where the
# CI fusion and its weight actually act. Baselines (omega=0.5, interval=3)
# included so every point carries the new column.
for omega in ["0.3", "0.5", "0.7", "trace"]:
    CONFIGS.append((f"A1b_omega{omega}", DEC, True, 3,
                    {"COMPASS_EVAL_CI_OMEGA": omega}))
for interval in ["1", "3", "5", "10", "999999"]:
    CONFIGS.append((f"A2b_comint{interval}", DEC, True, 3,
                    {"COMPASS_EVAL_COMM_INTERVAL": interval}))

# A5 — planner ablation: the classical AUCTION planner running on OUR grid
# GP-Kalman belief (centralized sensing, mirroring COMPASS's centralized GP).
# Compare against COMPASS-side AUCTION (same planner on per-target GP belief):
# isolates the representation's contribution with the planner held fixed.
for na in [2, 3, 5]:
    CONFIGS.append((f"A5_auction_grid_na{na}", CENT, False, na,
                    {"KFGP_EVAL_METHOD": "AUCTION"}))

DISPLAY_METRICS = ["avgunc", "minnvisit", "avgnvisit", "avggapvisit"]


def result_dir(tag):
    return os.path.join(PROJECT_ROOT, f"ablation_{tag}")


def metric_csv(tag):
    return os.path.join(result_dir(tag), "metric",
                        "tgt6_speed20_b30_g200_h100_metric.csv")


def run_config(tag, run_name, decentral, na, extra):
    if os.path.exists(metric_csv(tag)):
        print(f"skip {tag}: metric CSV exists")
        return 0
    env = os.environ.copy()
    env.update({
        "COMPASS_EVAL_RUN_NAME":      run_name,
        "COMPASS_EVAL_TARGET_SIZE":   "6",
        "COMPASS_EVAL_NUM_AGENTS":    str(na),
        "COMPASS_EVAL_DECENTRALIZED": "1" if decentral else "0",
        "COMPASS_EVAL_NO_COLLISION":  "1",
        "COMPASS_EVAL_RESULT_PATH":   result_dir(tag),
        **{f"COMPASS_EVAL_{k}": v for k, v in rb.FIXED.items()},
        **extra,
    })
    print(f"\n===== {tag}: model={run_name} na={na} decentral={decentral} {extra} =====",
          flush=True)
    return subprocess.run([sys.executable, EVAL_SCRIPT], env=env, cwd=PROJECT_ROOT).returncode


def read_stats(tag):
    path = metric_csv(tag)
    if not os.path.exists(path):
        return None
    cols = {m: [] for m in DISPLAY_METRICS}
    with open(path) as f:
        for row in csv.DictReader(f):
            for m in DISPLAY_METRICS:
                try:
                    cols[m].append(float(row[m]))
                except (KeyError, ValueError):
                    pass
    return {m: (float(np.mean(v)), 1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)))
            for m, v in cols.items() if v}


def main():
    rb.setup_shim()
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = rb.SHIM_DIR + (os.pathsep + old_pp if old_pp else "")

    failed = []
    try:
        for tag, run_name, decentral, na, extra in CONFIGS:
            rc = run_config(tag, run_name, decentral, na, extra)
            if rc != 0:
                failed.append(tag)
    finally:
        rb.teardown_shim()
        if old_pp:
            os.environ["PYTHONPATH"] = old_pp
        else:
            os.environ.pop("PYTHONPATH", None)

    print("\n\n===== Ablation sweep summary (mean ± 95% CI, n=32) =====")
    for tag, *_ in CONFIGS:
        stats = read_stats(tag)
        if stats is None:
            print(f"  {tag}: MISSING")
            continue
        line = "  ".join(f"{m}={v[0]:.4f}±{v[1]:.4f}" for m, v in stats.items())
        print(f"  {tag}: {line}")

    if failed:
        print(f"\n[WARNING] failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
