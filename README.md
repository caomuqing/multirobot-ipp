# Multi-robot Learning-based Informative Path Planning Using Spatio-Temporal Gaussian Process Kalman Filter

Simulation code for the IEEE Robotics and Automation Letters paper by Muqing Cao\*, Yunwoo Lee\*, Junbin Yuan, Lorenzo Schenk and Sebastian Scherer (\*equal contribution).

Instead of one Gaussian process per target, the robots keep a single grid belief over anonymous target presence, updated by a GP–Kalman filter:

- a measurement update over the cells of the camera footprint, with range-dependent noise;
- a GP-derived process update that inflates uncertainty over time as targets move;
- a diagonal-covariance approximation with fixed kernel correlations, which keeps the belief small enough to exchange.

In decentralized deployment each robot runs its own mapper and fuses the beliefs it receives with diagonal covariance intersection. An attention-based policy, trained with PPO, selects each robot's next node on a navigation graph.

The code builds on COMPASS ([Zhang, Wang and Sartoretti, MRS 2025](https://arxiv.org/abs/2507.16306)).

## Contents

| Path | Contents |
|---|---|
| `stamp_mapper_multi.py` | GP–Kalman grid mapper: camera measurement model, measurement and process updates, CI fusion |
| `stamp_mapper_fullcov.py` | Exact full-covariance mapper (ablation) |
| `env.py` | Multi-robot persistent-monitoring environment |
| `network.py` | Attention-based policy network |
| `driver.py`, `runner.py`, `worker.py` | PPO training with Ray |
| `arguments.py` | Training (`Arguments`) and evaluation (`ArgumentsEval`) settings |
| `evals/` | Evaluation driver and worker, including the AUCTION and COVERAGE baselines on the grid belief |
| `models/` | The two trained policies used in the paper |
| `run_*.py`, `open_loop_fidelity.py` | Scripts for the paper's experiments (below) |
| `gaussian_process.py`, `utils/` | GP wrapper and graph, target-motion and TSP utilities from COMPASS |

## Installation

Tested with Python 3.10.

```bash
pip install -r requirements-docker.txt
```

This installs CPU PyTorch; for CUDA wheels add `--extra-index-url https://download.pytorch.org/whl/cu121`. Alternatively, `docker compose build` builds a GPU image (see `docker-compose.yml`).

## Training

Set the options in `Arguments` (`arguments.py`): for example `run_name`, `num_agents`, `decentralized_sensing` (per-robot mappers with belief exchange), the GPU settings and the number of parallel Ray workers (`num_meta`). Then run

```bash
python driver.py
```

Checkpoints are saved to `models/<run_name>/` and TensorBoard logs to `runs/<run_name>/`.

## Evaluation

The trained policies are `models/centralized_nocollision_hgt003` (PROPOSED-Centralized) and `models/decentralized_nocollision_hgt003` (PROPOSED-Decentralized). To run one episode and save its frames and GIF:

```bash
python run_single_eval.py --run_name centralized_nocollision_hgt003 --num_agents 3 --decentralized 0
python run_single_eval.py --run_name decentralized_nocollision_hgt003 --num_agents 3 --decentralized 1
```

`python evals/eval_driver.py` evaluates with the settings in `ArgumentsEval`. Metrics are written per episode to `<result_path>/metric/*.csv` (`avgunc`: average uncertainty; `minnvisit`, `avgnvisit`: minimum and average visit counts).

## Paper experiments

| Script | Experiment |
|---|---|
| `run_batch_eval.py` | Main comparison (Fig. 2, Table I): both policies for 2, 3 and 5 robots, with centralized or decentralized sensing and with or without the non-collision constraint. The sweep is configured at the top of the file. |
| `run_ablation_sweeps.py` | Ablations of Sec. VI-F: CI weight, exchange interval, team size, full covariance, AUCTION on the grid belief, process update off |
| `open_loop_fidelity.py` | Diagonal vs. full-covariance mapper on identical measurement sequences |
| `run_final_experiments.py` | Uncertainty on the robots' own beliefs (Fig. 2) and the COVERAGE replay probe (Sec. VI-F). Its `check-replay` mode compares against the adapted COMPASS baseline, which is not included here. |
| `plot_batch_eval.py`, `read_metrics.py` | Plot and print the evaluation results |

The AUCTION and COVERAGE planners are selected with `KFGP_EVAL_METHOD=AUCTION` or `KFGP_EVAL_METHOD=COVERAGE`; the scripts above set this and the other evaluation options through environment variables.

## Citation

```bibtex
@article{cao2026multirobot,
  title   = {Multi-robot Learning-based Informative Path Planning Using Spatio-Temporal Gaussian Process Kalman Filter},
  author  = {Cao, Muqing and Lee, Yunwoo and Yuan, Junbin and Schenk, Lorenzo and Scherer, Sebastian},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The code is derived from COMPASS, copyright 2023 MARMot Lab @ NUS-ME.
