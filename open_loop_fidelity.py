#!/usr/bin/env python3
"""
Open-loop fidelity study: diagonal-approximation mapper vs. exact
full-covariance GP-Kalman mapper (reviewer-requested ablation, R6).

Both mappers are fed IDENTICAL measurement sequences from simulated random
sensing trajectories (same footprints, same noise draws are irrelevant since
measurement values enter both identically), so any divergence is attributable
purely to the covariance representation.

Reported per seed, then aggregated:
  - mean/max |mu_diag - mu_full| over cells and steps (posterior-mean error)
  - mean variance ratio diag/full and RMS variance error (calibration cost)
  - wall-clock per measurement update and per process update, both mappers
Usage: conda run -n compass_env python open_loop_fidelity.py
"""

import time

import numpy as np

from stamp_mapper_multi import STAMP_FR_GPF_IK
from stamp_mapper_fullcov import STAMP_FR_GPF_IK_FullCov

KW = dict(x_boundary=1.0, y_boundary=1.0, x_resolution=32, y_resolution=32,
          f_prior=0.0, v_prior=1.0, l=0.06, elevated_regions=[],
          flight_altitude=0.03, temporal_alpha=0.97)
CAM = dict(camera_pitch=55.0, camera_hfov=40.0, camera_vfov=45.0,
           flight_altitude=0.03, fov_size=0.2)
NOISE = dict(noise_std=0.1, regularization=0.0, max_condition_number=1e8,
             noise_distance_scale=50.0)

N_SEEDS = 10
N_STEPS = 60          # 2x the episode budget of 30
N_TARGETS = 6
ALPHA = 0.97


def run_seed(seed):
    rng = np.random.default_rng(seed)
    diag = STAMP_FR_GPF_IK(**KW)
    full = STAMP_FR_GPF_IK_FullCov(**KW)

    # Random-walk agent + moving targets, mirroring the benchmark's scales
    pos = rng.uniform(0.2, 0.8, 2)
    tgts = rng.uniform(0.0, 1.0, (N_TARGETS, 2))

    mu_err_mean, mu_err_max = [], []
    var_ratio, var_rmse = [], []
    t_diag_up, t_full_up = [], []
    t_diag_pr, t_full_pr = [], []

    for _ in range(N_STEPS):
        step_dir = rng.uniform(-np.pi, np.pi)
        pos = np.clip(pos + 0.08 * np.array([np.cos(step_dir), np.sin(step_dir)]), 0.0, 1.0)
        tgts = np.clip(tgts + rng.normal(0.0, 1.0 / 40.0, tgts.shape), 0.0, 1.0)
        heading = step_dir

        m, Hm, dist, _ = diag.take_measurement(pos, list(tgts), agent_heading=heading, **CAM)
        m2, Hm2, dist2, _ = full.take_measurement(pos, list(tgts), agent_heading=heading, **CAM)
        assert list(Hm) == list(Hm2)

        if len(Hm) > 0:
            t0 = time.perf_counter(); diag.update_map(m, Hm, distances=dist, **NOISE)
            t_diag_up.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); full.update_map(m2, Hm2, distances=dist2, **NOISE)
            t_full_up.append(time.perf_counter() - t0)

        t0 = time.perf_counter(); diag.process_update(alpha=ALPHA)
        t_diag_pr.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); full.process_update(alpha=ALPHA)
        t_full_pr.append(time.perf_counter() - t0)

        d_mu = np.abs(diag.state_value - full.state_value)
        mu_err_mean.append(d_mu.mean())
        mu_err_max.append(d_mu.max())
        var_ratio.append((diag.state_var / np.maximum(full.state_var, 1e-12)).mean())
        var_rmse.append(np.sqrt(((diag.state_var - full.state_var) ** 2).mean()))

    return dict(
        mu_err_mean=np.mean(mu_err_mean), mu_err_max=np.max(mu_err_max),
        var_ratio=np.mean(var_ratio), var_rmse=np.mean(var_rmse),
        t_diag_up=np.mean(t_diag_up) * 1e3, t_full_up=np.mean(t_full_up) * 1e3,
        t_diag_pr=np.mean(t_diag_pr) * 1e3, t_full_pr=np.mean(t_full_pr) * 1e3,
    )


def main():
    results = [run_seed(s) for s in range(N_SEEDS)]
    keys = results[0].keys()
    agg = {k: (np.mean([r[k] for r in results]), np.std([r[k] for r in results], ddof=1))
           for k in keys}

    print(f"Open-loop fidelity, {N_SEEDS} seeds x {N_STEPS} steps, 32x32 grid, "
          f"alpha={ALPHA}, l={KW['l']}")
    print(f"  posterior-mean error  |mu_d - mu_f| : mean {agg['mu_err_mean'][0]:.5f} "
          f"± {agg['mu_err_mean'][1]:.5f}, worst-cell max {agg['mu_err_max'][0]:.4f}")
    print(f"  variance ratio diag/full            : {agg['var_ratio'][0]:.4f} "
          f"± {agg['var_ratio'][1]:.4f}")
    print(f"  variance RMSE                       : {agg['var_rmse'][0]:.5f}")
    print(f"  measurement update  diag / full     : {agg['t_diag_up'][0]:.2f} ms / "
          f"{agg['t_full_up'][0]:.2f} ms")
    print(f"  process update      diag / full     : {agg['t_diag_pr'][0]:.3f} ms / "
          f"{agg['t_full_pr'][0]:.3f} ms")
    print(f"  belief memory       diag / full     : "
          f"{2 * 1024 * 8 / 1024:.0f} KB / {(1024 * 1024 + 2 * 1024) * 8 / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
