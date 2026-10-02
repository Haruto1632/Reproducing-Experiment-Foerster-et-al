"""
run_experiment.py
=================
Entrypoint for reproducing Foerster et al. 2016, Figure 4.

Usage:
    # Full n=3 experiment (5 seeds × 5 algorithms × 5000 epochs):
    python run_experiment.py --n 3

    # Full n=4 experiment:
    python run_experiment.py --n 4

    # Smoke test (2 seeds × 5 algorithms × 100 epochs):
    python run_experiment.py --n 3 --smoke

    # Single algorithm:
    python run_experiment.py --n 3 --alg dial --seeds 0 1 2

Records the following per run (per approved spec):
  - seed, n, algorithm, param_sharing
  - Python version, package versions
  - completed_episodes, update_steps, env_timesteps, plotted_x_axis
  - target_update_count, last_target_update_episode
  - mean_reward, std_reward at each eval checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

# Make the reproduction package importable from the repo root
# The switch_riddle package lives inside research_reproduction/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_this_dir, "..", "..")
sys.path.insert(0, _this_dir)           # for: import switch_riddle.*
sys.path.insert(0, os.path.join(_this_dir, ".."))  # for: import research_reproduction.*

from switch_riddle.training.trainer import Trainer, TrainingConfig, print_device_info, get_device


# ---------------------------------------------------------------------------
# Algorithm definitions
# ---------------------------------------------------------------------------

ALGORITHMS = [
    dict(algorithm="dial",   param_sharing=True,  label="DIAL"),
    dict(algorithm="dial",   param_sharing=False, label="DIAL-NS"),
    dict(algorithm="rial",   param_sharing=True,  label="RIAL"),
    dict(algorithm="rial",   param_sharing=False, label="RIAL-NS"),
    dict(algorithm="nocomm", param_sharing=True,  label="NoComm"),
]

N_EPOCHS = {3: 5000, 4: 40000}
EVAL_EVERY = {3: 50, 4: 200}


# ---------------------------------------------------------------------------
# Environment / package metadata
# ---------------------------------------------------------------------------

def collect_metadata() -> dict:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Switch Riddle reproduction experiment."
    )
    parser.add_argument("--n", type=int, choices=[3, 4], default=3,
                        help="Number of agents (3 or 4).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random seeds to run.")
    parser.add_argument("--alg", type=str, default=None,
                        help="Run only this algorithm (dial/rial/nocomm/dial-ns/rial-ns).")
    parser.add_argument("--smoke", action="store_true",
                        help="Short smoke test: 2 seeds, 100 epochs.")
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "results"),
                        help="Directory to write result logs.")
    args = parser.parse_args()

    n = args.n
    seeds = args.seeds[:2] if args.smoke else args.seeds
    max_epochs = 100 if args.smoke else N_EPOCHS[n]
    eval_every = 10 if args.smoke else EVAL_EVERY[n]
    eval_eps   = 32 if args.smoke else 200

    alg_filter = args.alg.lower().replace("-", "") if args.alg else None
    algs = [a for a in ALGORITHMS
            if alg_filter is None
            or a["label"].lower().replace("-", "") == alg_filter]

    os.makedirs(args.results_dir, exist_ok=True)

    meta = collect_metadata()
    meta["device"] = str(get_device())
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
    meta_path = os.path.join(args.results_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")
    print(f"Python:   {meta['python_version']}")
    print(f"PyTorch:  {meta['torch_version']}")
    print_device_info()
    print(f"\n{'='*60}")
    print(f"Experiment: n={n}  epochs={max_epochs}  seeds={seeds}")
    print(f"Algorithms: {[a['label'] for a in algs]}")
    print(f"{'='*60}\n")

    all_results = {}

    for alg_def in algs:
        label = alg_def["label"]
        all_results[label] = []

        for seed in seeds:
            print(f"\n--- {label}  n={n}  seed={seed} ---")
            cfg = TrainingConfig(
                n_agents       = n,
                algorithm      = alg_def["algorithm"],
                param_sharing  = alg_def["param_sharing"],
                algorithm_label= label,
                max_epochs     = max_epochs,
                eval_every_epochs = eval_every,
                eval_episodes  = eval_eps,
                seed           = seed,
                log_dir        = args.results_dir,
            )
            t0 = time.time()
            trainer = Trainer(cfg)
            history = trainer.train()
            elapsed = time.time() - t0

            if history:
                final = history[-1]
                print(
                    f"  Finished: mean_r={final['mean_reward']:.3f}  "
                    f"std={final['std_reward']:.3f}  "
                    f"elapsed={elapsed:.1f}s"
                )
            all_results[label].append({
                "seed": seed,
                "history": history,
                "elapsed_s": elapsed,
            })

    # Save aggregated results
    agg_path = os.path.join(args.results_dir,
                            f"aggregated_n{n}.json")
    with open(agg_path, "w") as f:
        json.dump({"n": n, "metadata": meta, "results": all_results}, f, indent=2)
    print(f"\nAggregated results: {agg_path}")


if __name__ == "__main__":
    main()
