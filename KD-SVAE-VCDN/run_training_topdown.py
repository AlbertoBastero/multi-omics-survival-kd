# -*- coding: utf-8 -*-
"""
Top-Down KD-SVAE-VCDN training pipeline.

Implements the top-down knowledge distillation hierarchy:
  Level 1: One teacher trained on all 3 modalities (complete cases)
  Level 2: Three pairwise teachers distilled from L1
  Level 3: Three single-modality students distilled from L2
  Integration: Fusion of the 3 students' latent representations

All levels train on complete-case patients (those with all 3 modalities),
using only the modalities relevant to that level.  This is the natural
consequence of the top-down design: the richest representation (all
modalities) is learned first and progressively distilled into simpler
single-modality models.

Usage (called from run_training.py with --architecture topdown):
    python run_training.py --architecture topdown
    python run_training.py --architecture topdown --cross_validation --n_folds 5

Author: Alberto Bastero
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from config import (
    device, num_class,
    layer_size_te1, layer_size_te2,
    latent_dim_te1, latent_dim_te2,
    topdown_l1_layers_size, topdown_l1_latent_dim,
)
from train_test import (
    to_categorical, get_predictions,
    plot_distillation_losses, plot_roc_curves_cv,
    plot_loss_with_ci, plot_accuracy_with_ci,
    plot_roc_curve_single, plot_training_curves,
)
from train_test_topdown import (
    training_topdown_l1, get_topdown_l1_predictions,
    training_topdown_l2, training_topdown_l3,
    training_integration, testing_topdown,
)


# Mapping from pair tag to modality indices (0-based)
PAIR_MOD_INDICES = {'12': (0, 1), '13': (0, 2), '23': (1, 2)}
PAIR_NAMES = {'12': 'miRNA+RNAseq', '13': 'miRNA+Meth', '23': 'RNAseq+Meth'}

# Which L2 teachers distil into each L3 student
# student mod_id → (pair_tag_a, pair_tag_b)
STUDENT_KD_SOURCES = {
    1: ('12', '13'),   # miRNA  ← L2(miRNA+RNAseq) + L2(miRNA+Meth)
    2: ('12', '23'),   # RNAseq ← L2(miRNA+RNAseq) + L2(RNAseq+Meth)
    3: ('13', '23'),   # Meth   ← L2(miRNA+Meth)   + L2(RNAseq+Meth)
}
MOD_NAMES = {1: 'miRNA', 2: 'RNAseq', 3: 'Methylation'}

HIGH_DIM_THRESHOLD = 8000


# =========================================================================
# CORE: train one fold (top-down)
# =========================================================================

def train_fold_topdown(kd_data, args, fold_idx=None):
    """Train the full top-down KD hierarchy on complete-case patients."""

    fold_str = f" (Fold {fold_idx + 1})" if fold_idx is not None else ""
    save_dir = (args.save_dir if fold_idx is None
                else os.path.join(args.save_dir, f'fold_{fold_idx + 1}'))
    os.makedirs(save_dir, exist_ok=True)

    # Architecture from config
    l1_layers = list(topdown_l1_layers_size)
    l1_ldim = topdown_l1_latent_dim
    layers_te1 = list(layer_size_te1)
    layers_te2 = list(layer_size_te2)
    ldim_te1 = latent_dim_te1
    ldim_te2 = latent_dim_te2

    focal_gamma = 2.0

    def class_weights_for(y):
        counts = np.bincount(y.astype(int), minlength=num_class)
        w = len(y) / (num_class * counts.astype(float) + 1e-8)
        return torch.tensor(w, dtype=torch.float32).to(device)

    # Complete-case data (used at every level)
    x1, x2, x3, y_complete, ids = kd_data['complete_train']
    cw = class_weights_for(y_complete)
    y_cat = to_categorical(y_complete, num_class).to(device)
    modalities = [x1, x2, x3]

    print(f"Complete-case patients (train): {len(y_complete)}")

    # ==================================================================
    # STEP 1 – L1 all-modality teacher
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 1: Training Top-Down L1 Teacher "
          f"(all modalities){fold_str}\n{'='*70}")

    training_topdown_l1(
        data1=x1, data2=x2, data3=x3,
        num_class=num_class, layers_size=l1_layers, lr=args.lr,
        latent_dim=l1_ldim, y=y_complete, y_cat=y_cat,
        num_epoch=args.te_epochs, batch_size=args.batch_size,
        temperature=args.temperature, save_dir=save_dir,
        class_weights=cw, gamma=focal_gamma,
        early_stop_acc=args.early_stop_acc,
        kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_cycle_length=args.kl_cycle_length,
        fusion_mode=args.fusion,
        regularization=args.regularization,
        reg_lambda_l1=args.reg_lambda_l1, reg_lambda_l2=args.reg_lambda_l2,
    )

    # Regenerate L1 soft labels in eval mode
    soft_labels_l1 = get_topdown_l1_predictions(
        x1, x2, x3,
        model_path=os.path.join(save_dir, 'topdown_l1_teacher.pt'),
        num_class=num_class, layers_size=l1_layers,
        latent_dim=l1_ldim, temperature=args.temperature,
        fusion_mode=args.fusion,
    )

    # ==================================================================
    # STEP 2 – L2 pairwise teachers (KD from L1)
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 2: Training Top-Down L2 Pairwise "
          f"Teachers{fold_str}\n{'='*70}")

    l2_arch = {}

    for pair_tag, (idx_a, idx_b) in PAIR_MOD_INDICES.items():
        x_combined = torch.cat([modalities[idx_a], modalities[idx_b]],
                               dim=1).to(device)
        combined_dim = x_combined.shape[1]

        if combined_dim > HIGH_DIM_THRESHOLD:
            l2_layers_use, l2_ldim_use = layers_te1, ldim_te1
            arch_label = f"REDUCED {layers_te1}, latent={ldim_te1}"
        else:
            l2_layers_use, l2_ldim_use = layers_te2, ldim_te2
            arch_label = f"FULL {layers_te2}, latent={ldim_te2}"
        l2_arch[pair_tag] = (l2_layers_use, l2_ldim_use)

        print(f"\n{'-'*50}\nL2 Teacher {pair_tag} ({PAIR_NAMES[pair_tag]}) – "
              f"{len(y_complete)} patients, {combined_dim} features{fold_str}"
              f"\n  Architecture: {arch_label}\n{'-'*50}")

        _, dl_losses = training_topdown_l2(
            data=x_combined, num_class=num_class,
            layers_size_te1=layers_te1, layers_size_te2=l2_layers_use,
            lr=args.lr, latent_dim_te1=ldim_te1, latent_dim_te2=l2_ldim_use,
            y=y_complete, y_cat=y_cat,
            soft_labels_l1=soft_labels_l1, kd_weight=1.0,
            num_epoch=args.te_epochs, batch_size=args.batch_size,
            temperature=args.temperature, pair_tag=pair_tag, save_dir=save_dir,
            class_weights=cw, gamma=focal_gamma,
            early_stop_acc=args.early_stop_acc,
            kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
            kl_warmup_epochs=args.kl_warmup_epochs,
            kl_cycle_length=args.kl_cycle_length,
            regularization=args.regularization,
            reg_lambda_l1=args.reg_lambda_l1,
            reg_lambda_l2=args.reg_lambda_l2,
        )

        if fold_idx is None:
            plot_distillation_losses(
                dl_losses,
                save_path=os.path.join(save_dir,
                                       f'td_distill_l2_{pair_tag}.png'),
                title=f"Distillation – L2 Teacher {PAIR_NAMES[pair_tag]}",
            )

    # ==================================================================
    # STEP 3 – L3 single-modality students (KD from 2 L2 teachers)
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 3: Training Top-Down L3 Students{fold_str}\n{'='*70}")

    # Regenerate L2 soft labels in eval mode for each pair
    l2_softs = {}
    for pair_tag, (idx_a, idx_b) in PAIR_MOD_INDICES.items():
        x_pair = torch.cat([modalities[idx_a], modalities[idx_b]], dim=1)
        l2_layers_use, l2_ldim_use = l2_arch[pair_tag]
        l2_softs[pair_tag] = get_predictions(
            x_pair,
            model_path=os.path.join(save_dir, f'topdown_l2_te_{pair_tag}.pt'),
            num_class=num_class, layers_size_te1=layers_te1,
            layers_size_te2=l2_layers_use, latent_dim_te1=ldim_te1,
            latent_dim_te2=l2_ldim_use, temperature=args.temperature, step=2,
        )

    all_l3_dl = {}
    for mod_id, (pair_a, pair_b) in STUDENT_KD_SOURCES.items():
        x_mod = modalities[mod_id - 1]
        softs_a = l2_softs[pair_a]
        softs_b = l2_softs[pair_b]

        kd_a = 0 if args.no_distillation else args.kd_a
        kd_b = 0 if args.no_distillation else args.kd_b

        print(f"\n{'-'*50}\nL3 Student {mod_id} ({MOD_NAMES[mod_id]}) – "
              f"{len(y_complete)} patients, {x_mod.shape[1]} features"
              f"\n  KD from L2({pair_a}) and L2({pair_b}){fold_str}\n{'-'*50}")

        dl_losses = training_topdown_l3(
            data=x_mod, num_class=num_class,
            layers_size_te1=layers_te1, layers_size_te2=layers_te2,
            lr=args.lr, latent_dim_te1=ldim_te1, latent_dim_te2=ldim_te2,
            y=y_complete, y_cat=y_cat,
            soft_labels_l2a=softs_a, soft_labels_l2b=softs_b,
            a=kd_a, b=kd_b,
            num_epoch=args.stu_epochs, batch_size=args.batch_size,
            temperature=args.temperature, mod_id=mod_id, save_dir=save_dir,
            class_weights=cw, gamma=focal_gamma,
            early_stop_acc=args.early_stop_acc,
            kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
            kl_warmup_epochs=args.kl_warmup_epochs,
            kl_cycle_length=args.kl_cycle_length,
            regularization=args.regularization,
            reg_lambda_l1=args.reg_lambda_l1,
            reg_lambda_l2=args.reg_lambda_l2,
        )
        all_l3_dl[mod_id] = dl_losses

        if fold_idx is None:
            plot_distillation_losses(
                dl_losses,
                save_path=os.path.join(save_dir,
                                       f'td_distill_l3_{mod_id}.png'),
                title=f"Distillation – L3 Student {MOD_NAMES[mod_id]}",
            )

    # ==================================================================
    # STEP 4 – Integration module
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 4: Training Integration Module{fold_str}\n{'='*70}")

    int_epochs = max(args.stu_epochs // 2, 10)

    hist_loss, hist_acc = training_integration(
        data1=x1, data2=x2, data3=x3, num_class=num_class,
        layers_size_te1=layers_te1, layers_size_te2=layers_te2,
        latent_dim_te1=ldim_te1, latent_dim_te2=ldim_te2,
        integration_layers_size=layers_te1,
        y=y_complete, y_cat=y_cat,
        lr=(args.stu_lr if args.stu_lr is not None else args.lr),
        num_epoch=int_epochs, batch_size=args.batch_size,
        temperature=args.temperature, save_dir=save_dir,
        class_weights=cw, gamma=focal_gamma,
        fusion_mode=args.fusion,
        regularization=args.regularization,
        reg_lambda_l1=args.reg_lambda_l1,
        reg_lambda_l2=args.reg_lambda_l2,
    )

    # Save distillation losses
    dl_save = os.path.join(save_dir, 'topdown_distillation_losses.json')
    with open(dl_save, 'w') as f:
        serializable = {}
        for k, v in all_l3_dl.items():
            serializable[str(k)] = {kk: [float(x) for x in vv]
                                    for kk, vv in v.items()}
        json.dump(serializable, f)

    # ==================================================================
    # STEP 5 – Evaluate on test set
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 5: Evaluating on "
          f"{'Validation' if fold_idx is not None else 'Test'} Set"
          f"{fold_str}\n{'='*70}")

    x1_te, x2_te, x3_te, y_te, te_ids = kd_data['complete_test']
    y_te_cat = to_categorical(y_te, num_class)

    print(f"Complete-case test patients: {len(y_te)}")

    results = testing_topdown(
        x1_test=x1_te, x2_test=x2_te, x3_test=x3_te,
        num_class=num_class,
        layers_size_te1=layers_te1, layers_size_te2=layers_te2,
        latent_dim_te1=ldim_te1, latent_dim_te2=ldim_te2,
        integration_layers_size=layers_te1,
        y_test=y_te, y_test_cat=y_te_cat,
        batch_size=args.batch_size, temperature=args.temperature,
        save_dir=save_dir, optimize_threshold=True,
        fusion_mode=args.fusion,
    )

    f1, auc, y_proba, balanced_acc, precision, recall, accuracy, opt_thr = results

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'y_proba': y_proba,
        'y_test': y_te,
        'optimal_threshold': opt_thr,
        'loss_history': hist_loss,
        'acc_history': hist_acc,
    }


# =========================================================================
# CROSS-VALIDATION
# =========================================================================

def run_cross_validation_topdown(args, kd_data):
    """K-fold CV for the top-down architecture."""
    from run_training import _build_fold_kd

    print(f"\n{'='*70}\n{args.n_folds}-FOLD CROSS-VALIDATION (TOP-DOWN)\n{'='*70}")

    _, _, _, y_train, ids_train = kd_data['complete_train']
    _, _, _, y_test, ids_test = kd_data['complete_test']

    all_ids = ids_train + ids_test
    all_y = np.concatenate([y_train, y_test])

    # Rebuild merged per-modality data for fold re-indexing
    preprocessed_dir = os.path.join(os.path.dirname(__file__),
                                    args.data_dir, 'preprocessed')
    from data_loader import load_preprocessed_data
    import pandas as pd
    raw = load_preprocessed_data(preprocessed_dir)

    merged = {}
    for mod in ('mirna', 'rnaseq', 'meth'):
        train_df, test_df = raw[f'{mod}_train'], raw[f'{mod}_test']
        merged_df = (pd.concat([train_df, test_df])
                     if train_df is not None and test_df is not None
                     else (train_df or test_df))

        lab_key_tr = f'labels_{mod}_train'
        lab_key_te = f'labels_{mod}_test'
        if raw.get(lab_key_tr) is None and mod == 'meth':
            lab_key_tr = 'labels_methylation_train'
        if raw.get(lab_key_te) is None and mod == 'meth':
            lab_key_te = 'labels_methylation_test'

        lab_tr, lab_te = raw.get(lab_key_tr), raw.get(lab_key_te)
        merged_lab = (pd.concat([lab_tr, lab_te], ignore_index=True)
                      if lab_tr is not None and lab_te is not None
                      else (lab_tr or lab_te))
        merged[mod] = (merged_df, merged_lab)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results, fold_roc_data, fold_losses, fold_accuracies = [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(all_y)), all_y)):
        print(f"\n{'#'*70}\n# FOLD {fold_idx+1}/{args.n_folds}\n{'#'*70}")

        train_set = set(all_ids[i] for i in train_idx)
        val_set = set(all_ids[i] for i in val_idx)
        fold_kd = _build_fold_kd(merged, train_set, val_set)

        res = train_fold_topdown(fold_kd, args, fold_idx=fold_idx)
        fold_results.append(res)
        fold_roc_data.append((res['y_test'], res['y_proba'], res['auc']))
        fold_losses.append(res['loss_history'])
        fold_accuracies.append(res['acc_history'])

        print(f"\nFold {fold_idx+1} Results "
              f"(threshold={res['optimal_threshold']:.3f}):")
        for m in ('accuracy', 'balanced_accuracy', 'f1', 'auc'):
            print(f"  {m}: {res[m]:.4f}")

    _print_and_save_cv_topdown(fold_results, fold_roc_data,
                               fold_losses, fold_accuracies, args)
    return fold_results


def _print_and_save_cv_topdown(fold_results, fold_roc_data,
                               fold_losses, fold_accuracies, args):
    """Print and save cross-validation summary for top-down architecture."""
    print(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY (TOP-DOWN)\n{'='*70}")
    metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall',
               'auc', 'optimal_threshold']
    cv = {}
    for m in metrics:
        vals = [r[m] for r in fold_results]
        cv[f'{m}_mean'] = np.mean(vals)
        cv[f'{m}_std'] = np.std(vals)
        print(f"  {m:20s}: {cv[f'{m}_mean']:.4f} +/- {cv[f'{m}_std']:.4f}")

    path = os.path.join(args.save_dir, 'cv_results.txt')
    with open(path, 'w') as f:
        f.write(f"Top-Down KD-SVAE-VCDN {args.n_folds}-Fold CV Results\n")
        distill_str = "DISABLED" if args.no_distillation else "ENABLED"
        f.write(f"Knowledge Distillation: {distill_str}\n")
        f.write(f"Architecture: top-down\n")
        f.write("(Metrics with optimal threshold per fold)\n")
        f.write("=" * 50 + "\n\n")
        for m in metrics:
            f.write(f"{m:20s}: {cv[f'{m}_mean']:.4f} +/- "
                    f"{cv[f'{m}_std']:.4f}\n")
        f.write("\n\nPer-Fold Results:\n" + "-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"\nFold {i+1} (threshold = {r['optimal_threshold']:.3f}):\n")
            for m in metrics:
                f.write(f"  {m}: {r[m]:.4f}\n")
    print(f"\nResults saved to: {path}")

    plot_roc_curves_cv(
        fold_roc_data,
        save_path=os.path.join(args.save_dir, 'roc_curves_cv.png'),
        title=f"ROC Curves – Top-Down {args.n_folds}-Fold CV")
    plot_loss_with_ci(
        fold_losses,
        save_path=os.path.join(args.save_dir, 'integration_loss_cv.png'),
        title=f"Integration Loss – Top-Down {args.n_folds}-Fold CV")
    plot_accuracy_with_ci(
        fold_accuracies,
        save_path=os.path.join(args.save_dir, 'integration_accuracy_cv.png'),
        title=f"Integration Accuracy – Top-Down {args.n_folds}-Fold CV")
