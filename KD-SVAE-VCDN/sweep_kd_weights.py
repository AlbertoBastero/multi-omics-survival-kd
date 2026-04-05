#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KD weight (a, b, c) sweep for KD-SVAE-VCDN.

Runs the training pipeline with different (kd_a, kd_b, kd_c) combinations,
then produces:
  1. A KL divergence plot across training epochs for each config (paper-style)
  2. A performance comparison bar chart
  3. A summary table identifying the best configuration

The parameters a, b, c control how much the student prioritises each
Level-2 teacher's knowledge:
  a  -> Teacher 12 (miRNA + RNAseq)
  b  -> Teacher 13 (miRNA + Methylation)
  c  -> Teacher 23 (RNAseq + Methylation)

Usage:
    python sweep_kd_weights.py
    python sweep_kd_weights.py --cross_validation --n_folds 5
    python sweep_kd_weights.py --configs "0,0,0;0.1,0.1,0.1;1,1,1"
"""

import os
import sys
import subprocess
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
# Default (a, b, c) configurations to sweep (similar to the paper)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_CONFIGS = [
    (0,    0,    0),
    (10,   0.1,  1),
    (1,    0.1,  10),
    (0.1,  10,   1),
    (0.1,  1,    10),
    (10,   1,    0.1),
    (1,    10,   0.1),
    (0.1,  0.1,  0.1),
    (1,    1,    1),
]


def parse_args():
    parser = argparse.ArgumentParser(description='KD weight sweep')
    parser.add_argument(
        '--configs', type=str, default=None,
        help='Semicolon-separated a,b,c triples. '
             'E.g. "0,0,0;0.1,0.1,0.1;1,10,0.1"')
    parser.add_argument('--base_save_dir', type=str, default='./sweep_kd_weights')
    # Training args forwarded to run_training.py
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--te_epochs', type=int, default=30)
    parser.add_argument('--stu_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stu_lr', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--early_stop_acc', type=float, default=None)
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)
    return parser.parse_args()


def parse_configs(config_str):
    """Parse "a,b,c;a,b,c;..." into list of tuples."""
    configs = []
    for triple in config_str.split(';'):
        parts = triple.strip().split(',')
        configs.append(tuple(float(x.strip()) for x in parts))
    return configs


def config_label(a, b, c):
    """Human-readable label for a config."""
    def fmt(v):
        return str(int(v)) if v == int(v) else str(v)
    return f"a={fmt(a)}, b={fmt(b)}, c={fmt(c)}"


def config_dirname(a, b, c):
    """Filesystem-safe directory name."""
    return f"a{a}_b{b}_c{c}"


# ─────────────────────────────────────────────────────────────────────
# Run one training
# ─────────────────────────────────────────────────────────────────────

def run_training(a, b, c, save_dir, args):
    cmd = [
        sys.executable, 'run_training.py',
        '--kd_a', str(a), '--kd_b', str(b), '--kd_c', str(c),
        '--save_dir', save_dir,
        '--data_dir', args.data_dir,
        '--batch_size', str(args.batch_size),
        '--te_epochs', str(args.te_epochs),
        '--stu_epochs', str(args.stu_epochs),
        '--lr', str(args.lr),
        '--temperature', str(args.temperature),
    ]
    if args.stu_lr is not None:
        cmd += ['--stu_lr', str(args.stu_lr)]
    if args.early_stop_acc is not None:
        cmd += ['--early_stop_acc', str(args.early_stop_acc)]
    if args.cross_validation:
        cmd += ['--cross_validation', '--n_folds', str(args.n_folds)]

    label = config_label(a, b, c)
    print(f"\n{'='*70}")
    print(f"  Running {label}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────
# Parse results
# ─────────────────────────────────────────────────────────────────────

def parse_results(results_path):
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


def load_distillation_losses(save_dir, is_cv=False, n_folds=5):
    """Load per-epoch distillation losses (dl1, dl2, dl3).

    For CV, average the per-fold distillation losses.
    Returns dict with keys 'dl1', 'dl2', 'dl3', each a list of floats.
    """
    if not is_cv:
        path = os.path.join(save_dir, 'distillation_losses_student.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # CV: average across folds
    all_fold_dl = []
    for fold_i in range(1, n_folds + 1):
        path = os.path.join(save_dir, f'fold_{fold_i}', 'distillation_losses_student.json')
        if os.path.exists(path):
            with open(path) as f:
                all_fold_dl.append(json.load(f))

    if not all_fold_dl:
        return None

    # Average across folds (pad shorter folds with last value)
    max_epochs = max(len(fd['dl1']) for fd in all_fold_dl)
    averaged = {}
    for key in ('dl1', 'dl2', 'dl3'):
        padded = []
        for fd in all_fold_dl:
            vals = fd[key]
            if len(vals) < max_epochs:
                vals = vals + [vals[-1]] * (max_epochs - len(vals))
            padded.append(vals)
        averaged[key] = np.mean(padded, axis=0).tolist()
    return averaged


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_kl_divergence(all_dl, configs, save_path):
    """Plot combined KL divergence (a*dl1 + b*dl2 + c*dl3) per epoch
    for each (a,b,c) configuration — reproduces the paper figure."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(configs), 10)))

    for i, (a, b, c) in enumerate(configs):
        key = config_dirname(a, b, c)
        if key not in all_dl or all_dl[key] is None:
            continue
        dl = all_dl[key]
        dl1 = np.array(dl['dl1'])
        dl2 = np.array(dl['dl2'])
        dl3 = np.array(dl['dl3'])
        # Combined weighted KL divergence
        combined = a * dl1 + b * dl2 + c * dl3
        epochs = np.arange(1, len(combined) + 1)
        ax.plot(epochs, combined, linewidth=1.8,
                label=config_label(a, b, c),
                color=colors[i % len(colors)])

    ax.set_xlabel('training epochs', fontsize=13)
    ax.set_ylabel('KL divergence', fontsize=13)
    ax.set_title('KL Divergence for Different KD Weight Configurations', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved KL divergence plot to: {save_path}")


def plot_performance_comparison(all_results, configs, save_path):
    """Bar chart comparing balanced accuracy across configs."""
    labels = []
    bal_accs = []
    stds = []

    for a, b, c in configs:
        key = config_dirname(a, b, c)
        if key not in all_results:
            continue
        r = all_results[key]
        labels.append(config_label(a, b, c))
        bal_accs.append(r.get('balanced_accuracy', 0))
        stds.append(r.get('balanced_accuracy_std', 0))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, bal_accs, yerr=stds if any(s > 0 for s in stds) else None,
                  capsize=4, color=plt.cm.tab10(np.linspace(0, 1, min(len(labels), 10))),
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Balanced Accuracy by KD Weights (a, b, c)', fontsize=14,
                 fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance plot to: {save_path}")


def plot_metrics_comparison(all_results, configs, save_path):
    """Paper-style line plot of all metrics across configs."""
    metric_keys = ['balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    metric_labels = ['balanced acc', 'F1-score', 'Precision', 'Recall', 'AUC']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_keys))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(configs), 10)))

    for i, (a, b, c) in enumerate(configs):
        key = config_dirname(a, b, c)
        if key not in all_results:
            continue
        r = all_results[key]
        vals = [r.get(k, 0) for k in metric_keys]
        ax.plot(x, vals, marker='o', linewidth=2, markersize=6,
                label=config_label(a, b, c), color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel('value', fontsize=12)
    ax.set_xlabel('Performances', fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison plot to: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.base_save_dir, exist_ok=True)

    configs = (parse_configs(args.configs) if args.configs
               else DEFAULT_CONFIGS)

    all_results = {}
    all_dl = {}

    for a, b, c in configs:
        dirname = config_dirname(a, b, c)
        save_dir = os.path.join(args.base_save_dir, dirname)

        if args.cross_validation:
            results_path = os.path.join(save_dir, 'cv_results.txt')
        else:
            results_path = os.path.join(save_dir, 'results.txt')

        # Skip if already completed
        if os.path.exists(results_path):
            label = config_label(a, b, c)
            print(f"{label} already done, loading from {results_path}")
            all_results[dirname] = parse_results(results_path)
            all_dl[dirname] = load_distillation_losses(
                save_dir, is_cv=args.cross_validation, n_folds=args.n_folds)
            continue

        success = run_training(a, b, c, save_dir, args)
        if success and os.path.exists(results_path):
            all_results[dirname] = parse_results(results_path)
            all_dl[dirname] = load_distillation_losses(
                save_dir, is_cv=args.cross_validation, n_folds=args.n_folds)
        else:
            print(f"WARNING: Training failed for {config_label(a, b, c)}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("KD WEIGHT SWEEP SUMMARY")
    print(f"{'='*90}")
    header = (f"{'Config':<28s} | {'Bal.Acc':>8s} | {'F1':>8s} | "
              f"{'Prec':>8s} | {'Recall':>8s} | {'AUC':>8s}")
    print(header)
    print('-' * len(header))

    best_key = None
    best_bal_acc = -1

    for a, b, c in configs:
        dirname = config_dirname(a, b, c)
        if dirname not in all_results:
            continue
        r = all_results[dirname]
        bal = r.get('balanced_accuracy', 0)
        std_str = ''
        if f'balanced_accuracy_std' in r:
            std_str = f" +/-{r['balanced_accuracy_std']:.4f}"
        print(f"{config_label(a, b, c):<28s} | {bal:8.4f}{std_str:>12s} | "
              f"{r.get('f1', 0):8.4f} | {r.get('precision', 0):8.4f} | "
              f"{r.get('recall', 0):8.4f} | {r.get('auc', 0):8.4f}")
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_key = (a, b, c)

    if best_key:
        print(f"\nBest config: {config_label(*best_key)}  "
              f"(balanced accuracy = {best_bal_acc:.4f})")

    # ── Save JSON ─────────────────────────────────────────────────
    summary = {}
    for a, b, c in configs:
        dirname = config_dirname(a, b, c)
        if dirname in all_results:
            summary[config_label(a, b, c)] = all_results[dirname]
    summary_path = os.path.join(args.base_save_dir, 'sweep_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_kl_divergence(
        all_dl, configs,
        os.path.join(args.base_save_dir, 'kl_divergence_comparison.png'))

    plot_performance_comparison(
        all_results, configs,
        os.path.join(args.base_save_dir, 'balanced_accuracy_comparison.png'))

    plot_metrics_comparison(
        all_results, configs,
        os.path.join(args.base_save_dir, 'metrics_comparison.png'))


if __name__ == '__main__':
    main()
