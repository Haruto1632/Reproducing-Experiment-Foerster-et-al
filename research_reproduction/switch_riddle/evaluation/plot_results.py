"""
Plot Results & Aggregate Analysis
==================================
Reads per-run JSON files from research_reproduction/switch_riddle/results/,
computes Oracle normalization, aggregates mean and standard deviation across
seeds for each algorithm, generates learning curve plots matching Figure 4,
and outputs a comparison summary.

Usage:
    python research_reproduction/switch_riddle/evaluation/plot_results.py --n 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def compute_oracle_expected_reward(n: int) -> float:
    """
    Computes the highest average reward achievable given access to the true state (Oracle).
    In Switch Riddle:
      - Horizon T = 4n - 6
      - The oracle policy tells immediately if and only if all n agents have visited the room.
      - If all n agents have visited, reward is +1.
      - If time horizon T is reached before all n agents have visited, reward is 0.
      - The expected reward of the Oracle is therefore exactly the probability that all
        n agents are chosen at least once in T draws uniformly at random with replacement.
      - Using inclusion-exclusion:
          P(all n in T) = sum_{k=0}^n (-1)^k * C(n, k) * ((n - k) / n)^T
    """
    T = 4 * n - 6
    prob = sum(
        ((-1) ** k) * math.comb(n, k) * (((n - k) / n) ** T)
        for k in range(n + 1)
    )
    return float(prob)


def load_runs(results_dir: str, n: int):
    """
    Finds all individual run files for a given n in results_dir.
    Matches files of the pattern: {ALGORITHM}_n{n}_seed{seed}.json
    """
    pattern = os.path.join(results_dir, f"*_n{n}_seed*.json")
    files = glob.glob(pattern)
    runs_by_alg = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        # Skip aggregated files
        if fname.startswith("aggregated_"):
            continue
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            alg = data.get("algorithm")
            if not alg:
                continue
            if alg not in runs_by_alg:
                runs_by_alg[alg] = []
            runs_by_alg[alg].append(data)
        except Exception as e:
            print(f"Warning: could not read {fpath}: {e}")

    return runs_by_alg


def analyze_and_plot(n: int, results_dir: str, output_plot_path: str = None):
    oracle_reward = compute_oracle_expected_reward(n)
    print(f"\n=======================================================")
    print(f"ANALYSIS FOR n = {n} (Time Horizon T = {4*n - 6})")
    print(f"Oracle Theoretical Expected Reward: {oracle_reward:.5f}")
    print(f"=======================================================\n")

    runs_by_alg = load_runs(results_dir, n)
    if not runs_by_alg:
        print(f"No run files found in {results_dir} for n={n}.")
        return

    # Standard styling and algorithm color/line map matching paper Figure 4
    # In Paper Fig 4:
    # DIAL: solid green/teal
    # DIAL-NS: dashed green/teal
    # RIAL: solid dark green/blue
    # RIAL-NS: dashed dark green/blue
    # NoComm: solid orange
    # Oracle: dashed gray/black horizontal line at 1.0
    alg_styles = {
        "DIAL":    {"color": "#1b9e77", "linestyle": "-",  "label": "DIAL"},
        "DIAL-NS": {"color": "#1b9e77", "linestyle": "--", "label": "DIAL-NS"},
        "RIAL":    {"color": "#7570b3", "linestyle": "-",  "label": "RIAL"},
        "RIAL-NS": {"color": "#7570b3", "linestyle": "--", "label": "RIAL-NS"},
        "NOCOMM":  {"color": "#d95f02", "linestyle": "-",  "label": "NoComm"},
        "NoComm":  {"color": "#d95f02", "linestyle": "-",  "label": "NoComm"},
    }

    plt.figure(figsize=(8, 6), dpi=150)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Oracle")

    summary_records = []

    for alg, runs in sorted(runs_by_alg.items()):
        seeds = [r["seed"] for r in runs]
        # Align history series by epoch/plotted_x_axis
        epochs_series = []
        rewards_series = []
        norm_rewards_series = []

        last_counters = []

        for r in runs:
            hist = r.get("history", [])
            if not hist:
                continue
            eps = [h["plotted_x_axis"] for h in hist]
            raw_r = [h["mean_reward"] for h in hist]
            norm_r = [h["mean_reward"] / oracle_reward for h in hist]
            epochs_series.append(eps)
            rewards_series.append(raw_r)
            norm_rewards_series.append(norm_r)
            last_counters.append(r.get("final_counters", {}))

        if not rewards_series:
            continue

        # Common epoch grid
        common_epochs = np.array(epochs_series[0])
        # Interpolate or match lengths
        min_len = min(len(s) for s in norm_rewards_series)
        common_epochs = common_epochs[:min_len]
        trimmed_norm = np.array([s[:min_len] for s in norm_rewards_series])
        trimmed_raw = np.array([s[:min_len] for s in rewards_series])

        mean_norm = np.mean(trimmed_norm, axis=0)
        std_norm = np.std(trimmed_norm, axis=0)

        final_raw_mean = np.mean([s[-1] for s in trimmed_raw])
        final_raw_std = np.std([s[-1] for s in trimmed_raw])
        final_norm_mean = np.mean([s[-1] for s in trimmed_norm])
        final_norm_std = np.std([s[-1] for s in trimmed_norm])

        avg_completed_eps = np.mean([c.get("completed_episodes", 0) for c in last_counters])
        avg_update_steps = np.mean([c.get("update_steps", 0) for c in last_counters])
        avg_env_timesteps = np.mean([c.get("env_timesteps", 0) for c in last_counters])
        avg_target_updates = np.mean([c.get("target_update_count", 0) for c in last_counters])

        style = alg_styles.get(alg, {"color": "black", "linestyle": "-", "label": alg})
        plt.plot(common_epochs, mean_norm, color=style["color"],
                 linestyle=style["linestyle"], linewidth=2.0, label=style["label"])
        plt.fill_between(common_epochs, mean_norm - std_norm, mean_norm + std_norm,
                         color=style["color"], alpha=0.15)

        summary_records.append({
            "algorithm": alg,
            "seeds": seeds,
            "completed_episodes": avg_completed_eps,
            "update_steps": avg_update_steps,
            "env_timesteps": avg_env_timesteps,
            "target_update_count": avg_target_updates,
            "final_raw_reward": f"{final_raw_mean:.3f} +/- {final_raw_std:.3f}",
            "final_norm_reward": f"{final_norm_mean:.3f} +/- {final_norm_std:.3f}",
            "final_norm_mean": final_norm_mean,
        })

    plt.xlabel("# Epochs", fontsize=12)
    plt.ylabel("Norm. R (Optimal)", fontsize=12)
    plt.title(f"Evaluation of n = {n} (Reproduction)", fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    if output_plot_path is None:
        output_plot_path = os.path.join(results_dir, f"figure4_n{n}_reproduction.png")
    plt.savefig(output_plot_path)
    print(f"Plot saved to: {output_plot_path}\n")

    # Print summary table
    print("--- EXPERIMENTAL SUMMARY ---")
    header = f"{'Algorithm':<10} | {'Seeds':<10} | {'Compl. Eps':<10} | {'Upd Steps':<10} | {'Raw Reward (Mean +/- Std)':<26} | {'Norm Reward (Mean +/- Std)':<26}"
    print(header)
    print("-" * len(header))
    for rec in summary_records:
        seed_str = str(rec["seeds"])
        print(f"{rec['algorithm']:<10} | {seed_str:<10} | {rec['completed_episodes']:<10.0f} | {rec['update_steps']:<10.0f} | {rec['final_raw_reward']:<26} | {rec['final_norm_reward']:<26}")

    # Save summary json
    summary_path = os.path.join(results_dir, f"summary_table_n{n}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\nSummary table saved to: {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "results"))
    parser.add_argument("--output_plot", type=str, default=None)
    args = parser.parse_args()
    analyze_and_plot(args.n, args.results_dir, args.output_plot)
