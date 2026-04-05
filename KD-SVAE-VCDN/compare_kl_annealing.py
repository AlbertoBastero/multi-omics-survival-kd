#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KL Annealing comparison for KD-SVAE-VCDN.

Runs the training pipeline with three KL annealing strategies (none, linear,
cyclical) and produces comparison plots of KLD, BCE, accuracy and total loss.

Usage:
    python compare_kl_annealing.py
    python compare_kl_annealing.py --kl_beta_max 0.2 --kl_warmup_epochs 15
    python compare_kl_annealing.py --cross_validation --n_folds 5
"""

import os
import sys
import subprocess
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


STRATEGIES = ['none', 'linear', 'cyclical']
STRATEGY_LABELS = {'none': 'Constant (no annealing)', 'linear': 'Linear', 'cyclical': 'Cyclical'}
STRATEGY_COLORS = {'none': '#1f77b4', 'linear': '#ff7f0e', 'cyclical': '#2ca02c'}


def parse_args():
    parser = argparse.ArgumentParser(description='KL Annealing comparison')
    parser.add_argument('--base_save_dir', type=str, default='./compare_kl_annealing',
                        help='Base directory for all comparison results')
    # KL annealing params
    parser.add_argument('--kl_beta_max', type=float, default=0.1)
    parser.add_argument('--kl_warmup_epochs', type=int, default=None)
    parser.add_argument('--kl_cycle_length', type=int, default=None)
    # Forward training args
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--te_epochs', type=int, default=30)
    parser.add_argument('--stu_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stu_lr', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--kd_a', type=float, default=1)
    parser.add_argument('--kd_b', type=float, default=10)
    parser.add_argument('--kd_c', type=float, default=0.1)
    parser.add_argument('--early_stop_acc', type=float, default=None)
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--no_distillation', action='store_true')
    return parser.parse_args()


def run_training(strategy, save_dir, args):
    """Run run_training.py with the given KL annealing strategy."""
    cmd = [
        sys.executable, 'run_training.py',
        '--kl_annealing', strategy,
        '--kl_beta_max', str(args.kl_beta_max),
        '--save_dir', save_dir,
        '--data_dir', args.data_dir,
        '--batch_size', str(args.batch_size),
        '--te_epochs', str(args.te_epochs),
        '--stu_epochs', str(args.stu_epochs),
        '--lr', str(args.lr),
        '--temperature', str(args.temperature),
        '--kd_a', str(args.kd_a),
        '--kd_b', str(args.kd_b),
        '--kd_c', str(args.kd_c),
    ]
    if args.kl_warmup_epochs is not None:
        cmd += ['--kl_warmup_epochs', str(args.kl_warmup_epochs)]
    if args.kl_cycle_length is not None:
        cmd += ['--kl_cycle_length', str(args.kl_cycle_length)]
    if args.stu_lr is not None:
        cmd += ['--stu_lr', str(args.stu_lr)]
    if args.early_stop_acc is not None:
        cmd += ['--early_stop_acc', str(args.early_stop_acc)]
    if args.no_distillation:
        cmd += ['--no_distillation']
    if args.cross_validation:
        cmd += ['--cross_validation', '--n_folds', str(args.n_folds)]

    print(f"\n{'='*70}")
    print(f"  Running KL annealing = {strategy}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def parse_results(results_path):
    """Parse results.txt or cv_results.txt into a dict of metrics."""
    metrics = {}
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Per-Fold Results' in line:
                break
            if ':' in line and not line.startswith('='):
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
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


def load_component_histories(save_dir):
    """Load component_histories.json from a run directory.

    In cross-validation mode the files live inside fold_X/ subdirectories.
    When multiple folds are found, the per-epoch values are averaged across
    folds (truncated to the shortest fold length).
    """
    # Single-run case
    single_path = os.path.join(save_dir, 'component_histories.json')
    if os.path.exists(single_path):
        with open(single_path, 'r') as f:
            return json.load(f)

    # Cross-validation case: look for fold_*/component_histories.json
    import glob
    fold_paths = sorted(glob.glob(os.path.join(save_dir, 'fold_*', 'component_histories.json')))
    if not fold_paths:
        return None

    fold_data = []
    for p in fold_paths:
        with open(p, 'r') as f:
            fold_data.append(json.load(f))

    # Average across folds, truncated to the shortest fold
    keys = fold_data[0].keys()
    min_len = min(len(fd[k]) for fd in fold_data for k in keys)
    averaged = {}
    for k in keys:
        stacked = np.array([fd[k][:min_len] for fd in fold_data])
        averaged[k] = stacked.mean(axis=0).tolist()
    return averaged


def plot_training_dynamics(all_histories, save_path):
    """Plot KLD, BCE, beta and total loss across strategies (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        ('kld', 'KL Divergence (raw, unweighted)', axes[0, 0]),
        ('bce', 'Reconstruction Loss (BCE)', axes[0, 1]),
        ('beta', r'$\beta$ (KL weight)', axes[1, 0]),
    ]

    for key, title, ax in panels:
        for strategy in STRATEGIES:
            hist = all_histories.get(strategy)
            if hist is None or key not in hist:
                continue
            epochs = np.arange(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], linewidth=2,
                    label=STRATEGY_LABELS[strategy],
                    color=STRATEGY_COLORS[strategy])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2)

    # Weighted KLD (beta * KLD) — this is what actually enters the loss
    ax = axes[1, 1]
    for strategy in STRATEGIES:
        hist = all_histories.get(strategy)
        if hist is None or 'kld' not in hist or 'beta' not in hist:
            continue
        weighted = np.array(hist['beta']) * np.array(hist['kld'])
        epochs = np.arange(1, len(weighted) + 1)
        ax.plot(epochs, weighted, linewidth=2,
                label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLORS[strategy])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\beta \times$ KLD')
    ax.set_title(r'Weighted KL Divergence ($\beta \times$ KLD)')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.suptitle('Student Training Dynamics — KL Annealing Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training dynamics plot to: {save_path}")


def plot_metrics_comparison(all_results, save_path):
    """Bar chart comparing final metrics across strategies."""
    metric_keys = ['balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    metric_labels = ['Balanced Acc', 'F1', 'Precision', 'Recall', 'AUC']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_keys))
    width = 0.25

    for i, strategy in enumerate(STRATEGIES):
        if strategy not in all_results:
            continue
        r = all_results[strategy]
        vals = [r.get(k, 0) for k in metric_keys]
        stds = [r.get(f'{k}_std', 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, yerr=stds if any(s > 0 for s in stds) else None,
                      label=STRATEGY_LABELS[strategy], color=STRATEGY_COLORS[strategy],
                      capsize=3, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_title('Final Metrics — KL Annealing Comparison')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison plot to: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.base_save_dir, exist_ok=True)

    all_results = {}
    all_histories = {}

    for strategy in STRATEGIES:
        save_dir = os.path.join(args.base_save_dir, f'kl_{strategy}')

        if args.cross_validation:
            results_path = os.path.join(save_dir, 'cv_results.txt')
        else:
            results_path = os.path.join(save_dir, 'results.txt')

        # Skip if already completed
        if os.path.exists(results_path):
            print(f"Strategy '{strategy}' already done, loading results.")
            all_results[strategy] = parse_results(results_path)
            hist = load_component_histories(save_dir)
            if hist:
                all_histories[strategy] = hist
            continue

        success = run_training(strategy, save_dir, args)
        if success and os.path.exists(results_path):
            all_results[strategy] = parse_results(results_path)
            hist = load_component_histories(save_dir)
            if hist:
                all_histories[strategy] = hist
        else:
            print(f"WARNING: Training failed for strategy '{strategy}'")

    # Print summary
    print(f"\n{'='*70}")
    print("KL ANNEALING COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Strategy':>12s} | {'Bal.Acc':>8s} | {'F1':>8s} | {'Prec':>8s} | {'Recall':>8s} | {'AUC':>8s}"
    print(header)
    print('-' * len(header))
    for strategy in STRATEGIES:
        if strategy in all_results:
            r = all_results[strategy]
            print(f"{strategy:>12s} | {r.get('balanced_accuracy', 0):8.4f} | "
                  f"{r.get('f1', 0):8.4f} | {r.get('precision', 0):8.4f} | "
                  f"{r.get('recall', 0):8.4f} | {r.get('auc', 0):8.4f}")

    # Save summary
    summary_path = os.path.join(args.base_save_dir, 'comparison_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    # Plot training dynamics (KLD, BCE, beta curves)
    if all_histories:
        plot_training_dynamics(
            all_histories,
            save_path=os.path.join(args.base_save_dir, 'training_dynamics.png'),
        )

    # Plot final metrics comparison
    if all_results:
        plot_metrics_comparison(
            all_results,
            save_path=os.path.join(args.base_save_dir, 'metrics_comparison.png'),
        )


if __name__ == '__main__':
    main()
