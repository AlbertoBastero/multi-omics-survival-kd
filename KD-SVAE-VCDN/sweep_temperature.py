#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature sweep for KD-SVAE-VCDN.
Runs the training pipeline with different temperature values and plots
a comparison of metrics (reproducing the style from Ranjbari et al.).

Usage:
    python sweep_temperature.py
    python sweep_temperature.py --temperatures 1.5 2.0 3.0 --stu_epochs 50
"""

import os
import sys
import subprocess
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Temperature sweep')
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                        help='Temperature values to sweep')
    parser.add_argument('--base_save_dir', type=str, default='./sweep_temperature',
                        help='Base directory for all sweep results')
    # Forward all other training args (use same defaults as run_training.py)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--te_epochs', type=int, default=30)
    parser.add_argument('--stu_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stu_lr', type=float, default=None)
    parser.add_argument('--kd_a', type=float, default=1)
    parser.add_argument('--kd_b', type=float, default=10)
    parser.add_argument('--kd_c', type=float, default=0.1)
    parser.add_argument('--early_stop_acc', type=float, default=None)
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)
    return parser.parse_args()


def run_training(temperature, save_dir, args):
    """Run run_training.py with the given temperature."""
    cmd = [
        sys.executable, 'run_training.py',
        '--temperature', str(temperature),
        '--save_dir', save_dir,
        '--data_dir', args.data_dir,
        '--batch_size', str(args.batch_size),
        '--te_epochs', str(args.te_epochs),
        '--stu_epochs', str(args.stu_epochs),
        '--lr', str(args.lr),
        '--kd_a', str(args.kd_a),
        '--kd_b', str(args.kd_b),
        '--kd_c', str(args.kd_c),
    ]
    if args.stu_lr is not None:
        cmd += ['--stu_lr', str(args.stu_lr)]
    if args.early_stop_acc is not None:
        cmd += ['--early_stop_acc', str(args.early_stop_acc)]
    if args.cross_validation:
        cmd += ['--cross_validation', '--n_folds', str(args.n_folds)]

    print(f"\n{'='*70}")
    print(f"  Running temperature = {temperature}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def parse_results(results_path):
    """Parse results.txt or cv_results.txt into a dict of metrics.

    Handles both formats:
      - Single run: "metric: 0.1234"
      - CV:         "metric              : 0.1234 +/- 0.0567"
    """
    metrics = {}
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Stop parsing when we hit the per-fold section
            if 'Per-Fold Results' in line:
                break
            if ':' in line and not line.startswith('='):
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                # Handle "0.1234 +/- 0.0567" format (CV)
                if '+/-' in val:
                    mean_str, std_str = val.split('+/-')
                    try:
                        metrics[key] = float(mean_str.strip())
                        metrics[f'{key}_std'] = float(std_str.strip())
                    except ValueError:
                        metrics[key] = val
                else:
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        metrics[key] = val
    return metrics


def plot_comparison(all_results, temperatures, save_path):
    """Plot metric comparison across temperatures (paper-style)."""
    metric_keys = ['balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    metric_labels = ['balanced acc', 'F1-score', 'Precision', 'Recall', 'AUC']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metric_keys))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(temperatures)))

    for i, temp in enumerate(temperatures):
        if temp not in all_results:
            continue
        vals = [all_results[temp].get(k, 0) for k in metric_keys]
        ax.plot(x, vals, marker='o', linewidth=2, markersize=6,
                label=f'temperature = {temp}', color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel('value', fontsize=12)
    ax.set_xlabel('Performances', fontsize=12)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.base_save_dir, exist_ok=True)

    all_results = {}

    for temp in args.temperatures:
        save_dir = os.path.join(args.base_save_dir, f'temp_{temp}')
        # CV writes cv_results.txt, single run writes results.txt
        if args.cross_validation:
            results_path = os.path.join(save_dir, 'cv_results.txt')
        else:
            results_path = os.path.join(save_dir, 'results.txt')

        # Skip if already completed
        if os.path.exists(results_path):
            print(f"Temperature {temp} already done, loading results from {results_path}")
            all_results[temp] = parse_results(results_path)
            continue

        success = run_training(temp, save_dir, args)
        if success and os.path.exists(results_path):
            all_results[temp] = parse_results(results_path)
        else:
            print(f"WARNING: Training failed for temperature={temp}")

    # Print summary table
    print(f"\n{'='*70}")
    print("TEMPERATURE SWEEP SUMMARY")
    print(f"{'='*70}")
    header = f"{'Temp':>6s} | {'Bal.Acc':>8s} | {'F1':>8s} | {'Prec':>8s} | {'Recall':>8s} | {'AUC':>8s}"
    print(header)
    print('-' * len(header))
    for temp in args.temperatures:
        if temp in all_results:
            r = all_results[temp]
            print(f"{temp:6.1f} | {r.get('balanced_accuracy', 0):8.4f} | "
                  f"{r.get('f1', 0):8.4f} | {r.get('precision', 0):8.4f} | "
                  f"{r.get('recall', 0):8.4f} | {r.get('auc', 0):8.4f}")

    # Save summary as JSON
    summary_path = os.path.join(args.base_save_dir, 'sweep_results.json')
    with open(summary_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Plot comparison
    plot_path = os.path.join(args.base_save_dir, 'temperature_comparison.png')
    plot_comparison(all_results, args.temperatures, plot_path)


if __name__ == '__main__':
    main()
