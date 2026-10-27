#!/usr/bin/env python3
"""
Run a single eval episode for one (run_name, num_agents, decentralized, nc)
configuration, saving frames + GIF to gifs/<run_name>/.

Reuses the sitecustomize shim from run_batch_eval.py to inject COMPASS_EVAL_*
overrides into eval_driver.py.

Usage:
    conda run -n compass_env python run_single_eval.py \
        --run_name decentralized_nocollision_hgt003 \
        --num_agents 2 \
        --decentralized 1 \
        --no_collision 1
"""

import argparse
import os
import subprocess
import sys

import run_batch_eval as rbe

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT  = os.path.join(PROJECT_ROOT, "evals", "eval_driver.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name",      default="decentralized_nocollision_hgt003")
    ap.add_argument("--target_size",   type=int, default=6)
    ap.add_argument("--num_agents",    type=int, default=2)
    ap.add_argument("--decentralized", type=int, default=1, choices=[0, 1])
    ap.add_argument("--no_collision",  type=int, default=1, choices=[0, 1])
    ap.add_argument("--budget_size",   type=int, default=30)
    ap.add_argument("--graph_size",    type=int, default=200)
    ap.add_argument("--history_size",  type=int, default=100)
    ap.add_argument("--target_speed",  type=float, default=1 / 20.0)
    ap.add_argument("--result_path",   default=None,
                    help="Defaults to <run_name>_single_eval")
    args = ap.parse_args()

    result_dir = args.result_path or os.path.join(
        PROJECT_ROOT, f"{args.run_name}_single_eval"
    )

    env = os.environ.copy()
    env.update({
        "COMPASS_EVAL_RUN_NAME":      args.run_name,
        "COMPASS_EVAL_TARGET_SIZE":   str(args.target_size),
        "COMPASS_EVAL_NUM_AGENTS":    str(args.num_agents),
        "COMPASS_EVAL_DECENTRALIZED": str(args.decentralized),
        "COMPASS_EVAL_NO_COLLISION":  str(args.no_collision),
        "COMPASS_EVAL_RESULT_PATH":   result_dir,
        "COMPASS_EVAL_NUM_EVAL":      "1",
        "COMPASS_EVAL_NUM_META":      "1",
        "COMPASS_EVAL_BUDGET_SIZE":   str(args.budget_size),
        "COMPASS_EVAL_GRAPH_SIZE":    str(args.graph_size),
        "COMPASS_EVAL_HISTORY_SIZE":  str(args.history_size),
        "COMPASS_EVAL_TARGET_SPEED":  str(args.target_speed),
        "COMPASS_EVAL_SAVE_IMG_GAP":  "1",
    })

    rbe.setup_shim()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = rbe.SHIM_DIR + (os.pathsep + old_pp if old_pp else "")

    print(f"Running single eval: run_name={args.run_name} na={args.num_agents} "
          f"decentral={args.decentralized} nc={args.no_collision}")
    print(f"Images → {PROJECT_ROOT}/gifs/{args.run_name}/")
    print(f"Metric → {result_dir}/metric/")

    try:
        ret = subprocess.run([sys.executable, EVAL_SCRIPT],
                             env=env, cwd=PROJECT_ROOT)
    finally:
        rbe.teardown_shim()

    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
