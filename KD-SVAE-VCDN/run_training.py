#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete training pipeline for KD-SVAE-VCDN.

Implements the 3-level knowledge distillation framework from Ranjbari et al.,
with correct per-modality patient subsets at each level:
  - Level 1 teachers: trained on all patients available for a single modality
  - Level 2 teachers: trained on the pairwise intersection of patients
  - Student: trained on complete-case patients (all 3 modalities)

Soft labels are re-generated between levels using the trained teacher on
the exact patient subset that the next level will use.

Usage:
    python run_training.py                              # single run, KD enabled
    python run_training.py --no_distillation            # single run, KD disabled
    python run_training.py --cross_validation --n_folds 5

Author: Alberto Bastero
"""

import os
import sys
import numpy as np
import torch
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    device, num_class,
    layer_size_te1, layer_size_te2,
    latent_dim_te1, latent_dim_te2,
    stu_layers_size, stu_latent_dim,
)
from data_loader import prepare_kd_data
from train_test import (
    training_te_level1,
    training_te_level2,
    training_stu,
    testing_stu,
    get_predictions,
    to_categorical,
    plot_distillation_losses,
    plot_roc_curves_cv,
    plot_loss_with_ci,
    plot_accuracy_with_ci,
    plot_roc_curve_single,
    plot_training_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description='KD-SVAE-VCDN Training')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--te_epochs', type=int, default=30)
    parser.add_argument('--stu_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for teachers')
    parser.add_argument('--stu_lr', type=float, default=0.0001,
                        help='Learning rate for student (defaults to --lr if not set)')
    parser.add_argument('--temperature', type=float, default=2.0)

    # KD control
    parser.add_argument('--no_distillation', action='store_true',
                        help='Disable knowledge distillation (student trains alone)')
    parser.add_argument('--kd_a', type=float, default=1, help='KD weight for teacher 1+2')
    parser.add_argument('--kd_b', type=float, default=10, help='KD weight for teacher 1+3')
    parser.add_argument('--kd_c', type=float, default=0.1, help='KD weight for teacher 2+3')

    # KL annealing
    parser.add_argument('--kl_annealing', type=str, default='none',
                        choices=['none', 'linear', 'cyclical'],
                        help='KL annealing strategy: none (constant), linear, or cyclical')
    parser.add_argument('--kl_beta_max', type=float, default=0.1,
                        help='Maximum beta value for KL term (default: 0.1)')
    parser.add_argument('--kl_warmup_epochs', type=int, default=None,
                        help='Warmup epochs for linear annealing (default: num_epochs // 2)')
    parser.add_argument('--kl_cycle_length', type=int, default=None,
                        help='Cycle length for cyclical annealing (default: num_epochs // 4)')

    # Student fusion mode
    parser.add_argument('--fusion', type=str, default='vcdn',
                        choices=['vcdn', 'concat'],
                        help='Student fusion strategy: vcdn (outer product) or concat (concatenation)')

    # Regularization
    parser.add_argument('--regularization', type=str, default='none',
                        choices=['none', 'l1', 'l2', 'elastic'],
                        help='Weight regularization: none, l1, l2, or elastic (l1+l2)')
    parser.add_argument('--reg_lambda_l1', type=float, default=1e-5,
                        help='L1 regularization coefficient (default: 1e-5)')
    parser.add_argument('--reg_lambda_l2', type=float, default=1e-4,
                        help='L2 regularization coefficient (default: 1e-4)')

    # CV / early stopping
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--early_stop_acc', type=float, default=None)

    return parser.parse_args()


# =========================================================================
# CORE: train one fold with correct per-level patient subsets
# =========================================================================

def train_fold(kd_data, args, fold_idx=None, split_suffix=''):
    """
    Train the full KD hierarchy on correctly separated patient subsets.

    kd_data is the dict returned by prepare_kd_data (or a re-indexed
    version for cross-validation).
    """
    fold_str = f" (Fold {fold_idx + 1})" if fold_idx is not None else ""
    save_dir = (args.save_dir if fold_idx is None
                else os.path.join(args.save_dir, f'fold_{fold_idx + 1}'))
    os.makedirs(save_dir, exist_ok=True)

    # Use architecture from config.py
    layers_te1 = list(layer_size_te1)
    layers_te2 = list(layer_size_te2)
    layers_stu = list(stu_layers_size)
    ldim_te1 = latent_dim_te1
    ldim_te2 = latent_dim_te2
    ldim_stu = stu_latent_dim

    # ------------------------------------------------------------------
    # helper: compute class weights for a label array
    # ------------------------------------------------------------------
    def class_weights_for(y):
        counts = np.bincount(y.astype(int), minlength=num_class)
        w = len(y) / (num_class * counts.astype(float) + 1e-8)
        return torch.tensor(w, dtype=torch.float32).to(device)

    focal_gamma = 2.0

    # ==================================================================
    # STEP 1 – Level 1 teachers (single modality, full per-modality set)
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 1: Training Level 1 Teachers{fold_str}\n{'='*70}")

    softs_per_teacher = {}  # teacher_id -> softs on its own training set

    for te_id, mod_name in [(1, 'miRNA'), (2, 'RNAseq'), (3, 'Methylation')]:
        data_tensor, y_arr, _ = kd_data[f'single_{te_id}_train']
        cw = class_weights_for(y_arr)
        y_cat = to_categorical(y_arr, num_class).to(device)

        print(f"\n{'-'*50}\nTeacher {te_id} ({mod_name}) – {len(y_arr)} patients, "
              f"{data_tensor.shape[1]} features{fold_str}\n{'-'*50}")

        softs = training_te_level1(
            data=data_tensor, num_class=num_class,
            layers_size_te1=layers_te1, layers_size_te2=layers_te2,
            lr=args.lr, latent_dim_te1=ldim_te1, latent_dim_te2=ldim_te2,
            y=y_arr, y_cat=y_cat, num_epoch=args.te_epochs,
            batch_size=args.batch_size, temperature=args.temperature,
            use_sample_weight=True, steps=1, num_te=te_id, save_dir=save_dir,
            class_weights=cw, gamma=focal_gamma,
            early_stop_acc=args.early_stop_acc,
            kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
            kl_warmup_epochs=args.kl_warmup_epochs, kl_cycle_length=args.kl_cycle_length,
            regularization=args.regularization, reg_lambda_l1=args.reg_lambda_l1,
            reg_lambda_l2=args.reg_lambda_l2,
        )
        softs_per_teacher[te_id] = softs

    # ==================================================================
    # STEP 2 – Level 2 teachers (pairwise, with KD from level 1)
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 2: Training Level 2 Teachers{fold_str}\n{'='*70}")

    pair_cfg = [
        # (pair_tag, te_id_a, te_id_b, l2_num, kd_weight_a, kd_weight_b)
        ('12', 1, 2, 1, 0.7, 0.3),
        ('13', 1, 3, 2, 0.3, 0.7),
        ('23', 2, 3, 3, 0.7, 0.3),
    ]
    pair_names = {'12': 'miRNA+RNAseq', '13': 'miRNA+Meth', '23': 'RNAseq+Meth'}

    softs_l2 = {}  # pair_tag -> softs on pairwise training set
    l2_arch = {}   # pair_tag -> (layers, latent_dim) used for each L2 teacher

    HIGH_DIM_THRESHOLD = 8000

    for pair_tag, tid_a, tid_b, l2_num, wa, wb in pair_cfg:
        data_a, data_b, y_pair, pair_ids = kd_data[f'pair_{pair_tag}_train']
        cw = class_weights_for(y_pair)
        y_cat = to_categorical(y_pair, num_class).to(device)

        softs_a = get_predictions(
            data_a, model_path=os.path.join(save_dir, f'KD_TE_1{tid_a}.pt'),
            num_class=num_class, layers_size_te1=layers_te1,
            layers_size_te2=layers_te2, latent_dim_te1=ldim_te1,
            latent_dim_te2=ldim_te2, temperature=args.temperature, step=1,
        )
        softs_b = get_predictions(
            data_b, model_path=os.path.join(save_dir, f'KD_TE_1{tid_b}.pt'),
            num_class=num_class, layers_size_te1=layers_te1,
            layers_size_te2=layers_te2, latent_dim_te1=ldim_te1,
            latent_dim_te2=ldim_te2, temperature=args.temperature, step=1,
        )

        x_combined = torch.cat([data_a, data_b], dim=1).to(device)
        combined_dim = x_combined.shape[1]

        # Adaptive architecture: use smaller L1 architecture for
        # high-dimensional combined inputs to avoid overparameterization
        # (the L1 architecture already works for 15k-dim methylation data)
        if combined_dim > HIGH_DIM_THRESHOLD:
            l2_layers_use = layers_te1
            l2_ldim_use = ldim_te1
            arch_label = f"REDUCED {layers_te1}, latent={ldim_te1}"
        else:
            l2_layers_use = layers_te2
            l2_ldim_use = ldim_te2
            arch_label = f"FULL {layers_te2}, latent={ldim_te2}"
        l2_arch[pair_tag] = (l2_layers_use, l2_ldim_use)

        print(f"\n{'-'*50}\nLevel 2 Teacher {l2_num} ({pair_names[pair_tag]}) – "
              f"{len(y_pair)} patients, {combined_dim} features{fold_str}"
              f"\n  Architecture: {arch_label}\n{'-'*50}")

        softs_pair, dl_losses = training_te_level2(
            data=x_combined, num_class=num_class,
            layers_size_te1=layers_te1, layers_size_te2=l2_layers_use,
            lr=args.lr, latent_dim_te1=ldim_te1, latent_dim_te2=l2_ldim_use,
            y=y_pair, y_cat=y_cat, softs1=softs_a, softs2=softs_b,
            a=wa, b=wb, num_epoch=args.te_epochs, batch_size=args.batch_size,
            temperature=args.temperature, use_sample_weight=True,
            steps=2, num_te=l2_num, save_dir=save_dir,
            class_weights=cw, gamma=focal_gamma,
            early_stop_acc=args.early_stop_acc,
            kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
            kl_warmup_epochs=args.kl_warmup_epochs, kl_cycle_length=args.kl_cycle_length,
            regularization=args.regularization, reg_lambda_l1=args.reg_lambda_l1,
            reg_lambda_l2=args.reg_lambda_l2,
        )
        softs_l2[pair_tag] = softs_pair

        if fold_idx is None:
            plot_distillation_losses(
                dl_losses,
                save_path=os.path.join(save_dir, f'distillation_loss_te_{pair_tag}.png'),
                title=f"Distillation Losses – L2 Teacher {pair_names[pair_tag]}",
            )

    # ==================================================================
    # STEP 3 – Student (complete cases, with KD from level 2)
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 3: Training Student Model{fold_str}\n{'='*70}")

    x1_stu, x2_stu, x3_stu, y_stu, stu_ids = kd_data['complete_train']
    cw = class_weights_for(y_stu)
    y_cat_stu = to_categorical(y_stu, num_class).to(device)

    print(f"Complete-case patients: {len(y_stu)}")

    # Re-generate L2 soft labels on complete-case patients
    # Use the same architecture that each L2 teacher was trained with
    l2_layers_12, l2_ldim_12 = l2_arch['12']
    l2_layers_13, l2_ldim_13 = l2_arch['13']
    l2_layers_23, l2_ldim_23 = l2_arch['23']

    x_12_stu = torch.cat([x1_stu, x2_stu], dim=1)
    softs_12_stu = get_predictions(
        x_12_stu, model_path=os.path.join(save_dir, 'KD_TE_21.pt'),
        num_class=num_class, layers_size_te1=layers_te1,
        layers_size_te2=l2_layers_12, latent_dim_te1=ldim_te1,
        latent_dim_te2=l2_ldim_12, temperature=args.temperature, step=2,
    )
    x_13_stu = torch.cat([x1_stu, x3_stu], dim=1)
    softs_13_stu = get_predictions(
        x_13_stu, model_path=os.path.join(save_dir, 'KD_TE_22.pt'),
        num_class=num_class, layers_size_te1=layers_te1,
        layers_size_te2=l2_layers_13, latent_dim_te1=ldim_te1,
        latent_dim_te2=l2_ldim_13, temperature=args.temperature, step=2,
    )
    x_23_stu = torch.cat([x2_stu, x3_stu], dim=1)
    softs_23_stu = get_predictions(
        x_23_stu, model_path=os.path.join(save_dir, 'KD_TE_23.pt'),
        num_class=num_class, layers_size_te1=layers_te1,
        layers_size_te2=l2_layers_23, latent_dim_te1=ldim_te1,
        latent_dim_te2=l2_ldim_23, temperature=args.temperature, step=2,
    )

    kd_a = 0 if args.no_distillation else args.kd_a
    kd_b = 0 if args.no_distillation else args.kd_b
    kd_c = 0 if args.no_distillation else args.kd_c

    hist_loss, hist_acc, conf, dl_losses_stu, component_histories = training_stu(
        data1=x1_stu, data2=x2_stu, data3=x3_stu,
        num_class=num_class, layers_size=layers_stu, lr=(args.stu_lr if args.stu_lr is not None else args.lr),
        stu_latent_dim=ldim_stu, y=y_stu, y_cat=y_cat_stu,
        softs1=softs_12_stu, softs2=softs_13_stu, softs3=softs_23_stu,
        a=kd_a, b=kd_b, c=kd_c,
        num_epoch=args.stu_epochs, batch_size=args.batch_size,
        temperature=args.temperature, use_sample_weight=True,
        save_dir=save_dir, class_weights=cw, gamma=focal_gamma,
        early_stop_acc=args.early_stop_acc,
        kl_annealing=args.kl_annealing, kl_beta_max=args.kl_beta_max,
        kl_warmup_epochs=args.kl_warmup_epochs, kl_cycle_length=args.kl_cycle_length,
        fusion_mode=args.fusion,
        regularization=args.regularization, reg_lambda_l1=args.reg_lambda_l1,
        reg_lambda_l2=args.reg_lambda_l2,
    )

    # Save distillation losses and component histories as JSON for analysis scripts
    import json as _json
    dl_save_path = os.path.join(save_dir, 'distillation_losses_student.json')
    with open(dl_save_path, 'w') as _f:
        _json.dump(dl_losses_stu, _f)
    comp_save_path = os.path.join(save_dir, 'component_histories.json')
    with open(comp_save_path, 'w') as _f:
        _json.dump(component_histories, _f)

    if fold_idx is None:
        plot_distillation_losses(
            dl_losses_stu,
            save_path=os.path.join(save_dir, 'distillation_loss_student.png'),
            title="Distillation Losses – Student Model",
        )

    # ==================================================================
    # STEP 4 – Evaluate on test complete-case patients
    # ==================================================================
    print(f"\n{'='*70}\nSTEP 4: Evaluating on "
          f"{'Validation' if fold_idx is not None else 'Test'} Set{fold_str}\n{'='*70}")

    x1_te, x2_te, x3_te, y_te, te_ids = kd_data['complete_test']
    y_te_cat = to_categorical(y_te, num_class)

    print(f"Complete-case test patients: {len(y_te)}")

    results = testing_stu(
        x1_test=x1_te, x2_test=x2_te, x3_test=x3_te,
        num_class=num_class, layers_size=layers_stu,
        latent_dim=ldim_stu, y_test=y_te, y_test_cat=y_te_cat,
        batch_size=args.batch_size,
        model_path=os.path.join(save_dir, 'brc_stu.pt'),
        optimize_threshold=True,
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
        'distillation_losses': dl_losses_stu,
        'component_histories': component_histories,
    }


# =========================================================================
# CROSS-VALIDATION (re-splits raw per-modality data)
# =========================================================================

def run_cross_validation(args, kd_data):
    """K-fold CV with correct per-fold KD subsets."""
    print(f"\n{'='*70}\n{args.n_folds}-FOLD CROSS-VALIDATION\n{'='*70}")

    # Use all single-modality patients (union) to drive the fold split.
    # The fold is defined on complete-case patients so train/val labels are clean.
    _, _, _, y_all_complete_train, ids_complete_train = kd_data['complete_train']
    _, _, _, y_all_complete_test, ids_complete_test = kd_data['complete_test']

    all_complete_ids = ids_complete_train + ids_complete_test
    all_complete_y = np.concatenate([y_all_complete_train, y_all_complete_test])

    # We also need per-modality data frames to re-index. Rebuild from raw.
    preprocessed_dir = os.path.join(os.path.dirname(__file__), args.data_dir, 'preprocessed')
    from data_loader import load_preprocessed_data, _extract_patient_ids_and_labels
    raw = load_preprocessed_data(preprocessed_dir)

    # Merge train+test per modality
    merged = {}
    for mod in ('mirna', 'rnaseq', 'meth'):
        import pandas as pd
        train_df = raw[f'{mod}_train']
        test_df = raw[f'{mod}_test']
        if train_df is not None and test_df is not None:
            merged_df = pd.concat([train_df, test_df])
        elif train_df is not None:
            merged_df = train_df
        else:
            merged_df = test_df

        lab_key_train = f'labels_{mod}_train'
        lab_key_test = f'labels_{mod}_test'
        # handle 'methylation' vs 'meth'
        if raw.get(lab_key_train) is None:
            lab_key_train = f'labels_methylation_train' if mod == 'meth' else lab_key_train
        if raw.get(lab_key_test) is None:
            lab_key_test = f'labels_methylation_test' if mod == 'meth' else lab_key_test

        lab_train = raw.get(lab_key_train)
        lab_test = raw.get(lab_key_test)
        if lab_train is not None and lab_test is not None:
            merged_lab = pd.concat([lab_train, lab_test], ignore_index=True)
        elif lab_train is not None:
            merged_lab = lab_train
        else:
            merged_lab = lab_test

        merged[mod] = (merged_df, merged_lab)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_roc_data = []
    fold_losses = []
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
            np.zeros(len(all_complete_y)), all_complete_y)):
        print(f"\n{'#'*70}\n# FOLD {fold_idx+1}/{args.n_folds}\n{'#'*70}")

        train_ids_fold = [all_complete_ids[i] for i in train_idx]
        val_ids_fold = [all_complete_ids[i] for i in val_idx]
        train_set = set(train_ids_fold)
        val_set = set(val_ids_fold)

        # Build kd_data-like dict for this fold
        fold_kd = _build_fold_kd(merged, train_set, val_set)

        results = train_fold(fold_kd, args, fold_idx=fold_idx)
        fold_results.append(results)
        fold_roc_data.append((results['y_test'], results['y_proba'], results['auc']))
        fold_losses.append(results['loss_history'])
        fold_accuracies.append(results['acc_history'])

        print(f"\nFold {fold_idx+1} Results (threshold={results['optimal_threshold']:.3f}):")
        for m in ('accuracy', 'balanced_accuracy', 'f1', 'auc'):
            print(f"  {m}: {results[m]:.4f}")

    # Aggregate
    _print_and_save_cv(fold_results, fold_roc_data, fold_losses, fold_accuracies, args)
    return fold_results


def _build_fold_kd(merged, train_set, val_set):
    """Build a kd_data dict for one CV fold from merged per-modality data.

    Re-applies MinMax scaling per fold (fit on train, transform both)
    so that all feature values are in [0, 1] for BCE reconstruction loss.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    result = {}
    mod_names = {1: 'mirna', 2: 'rnaseq', 3: 'meth'}

    # --- First pass: split each modality into train/val and fit scalers ---
    split_data = {}  # (mod_idx, split) -> (df, y, ids)

    for mod_idx, mod_name in mod_names.items():
        full_df, full_lab = merged[mod_name]
        if full_lab is None or full_df is None:
            continue
        lab = full_lab.copy()
        if 'patient_id' in lab.columns:
            lab = lab.set_index('patient_id')
        elif 'index' in lab.columns:
            lab = lab.set_index('index')
        else:
            lab = lab.set_index(lab.columns[0])

        # Get train and val patient IDs for this modality
        train_ids = sorted(set(full_df.index) & set(lab.index) & train_set)
        val_ids = sorted(set(full_df.index) & set(lab.index) & val_set)

        # Fit MinMax scaler on this fold's training data only
        scaler = MinMaxScaler()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(full_df.loc[train_ids].values),
            index=train_ids, columns=full_df.columns,
        )
        val_scaled = pd.DataFrame(
            scaler.transform(full_df.loc[val_ids].values).clip(0, 1),
            index=val_ids, columns=full_df.columns,
        )

        for split, ids, df_scaled in [('train', train_ids, train_scaled),
                                       ('test', val_ids, val_scaled)]:
            y = lab.loc[ids, 'label'].values.astype(int)
            split_data[(mod_idx, split)] = (df_scaled, y, ids)

            result[f'single_{mod_idx}_{split}'] = (
                torch.FloatTensor(df_scaled.values), y, ids,
            )

    # --- Pairwise and complete-case subsets ---
    for split in ('train', 'test'):
        dfs = {}
        ys = {}
        ids_per_mod = {}
        for mod_idx in mod_names:
            if (mod_idx, split) in split_data:
                df_s, y_s, id_s = split_data[(mod_idx, split)]
                dfs[mod_idx] = df_s
                ys[mod_idx] = y_s
                ids_per_mod[mod_idx] = id_s

        # Pairwise
        for tag, a, b in [('12', 1, 2), ('13', 1, 3), ('23', 2, 3)]:
            common = sorted(set(ids_per_mod.get(a, [])) & set(ids_per_mod.get(b, [])))
            if len(common) == 0:
                result[f'pair_{tag}_{split}'] = (
                    torch.zeros(0), torch.zeros(0), np.array([]), [])
                continue
            a_vals = dfs[a].loc[common].values
            b_vals = dfs[b].loc[common].values
            lab_a = merged[mod_names[a]][1].copy()
            if 'patient_id' in lab_a.columns:
                lab_a = lab_a.set_index('patient_id')
            elif 'index' in lab_a.columns:
                lab_a = lab_a.set_index('index')
            else:
                lab_a = lab_a.set_index(lab_a.columns[0])
            y_c = lab_a.loc[common, 'label'].values.astype(int)
            result[f'pair_{tag}_{split}'] = (
                torch.FloatTensor(a_vals),
                torch.FloatTensor(b_vals),
                y_c,
                common,
            )

        # Complete case
        complete = sorted(
            set(ids_per_mod.get(1, []))
            & set(ids_per_mod.get(2, []))
            & set(ids_per_mod.get(3, []))
        )
        if len(complete) > 0:
            lab_src = merged['rnaseq'][1].copy()
            if 'patient_id' in lab_src.columns:
                lab_src = lab_src.set_index('patient_id')
            elif 'index' in lab_src.columns:
                lab_src = lab_src.set_index('index')
            else:
                lab_src = lab_src.set_index(lab_src.columns[0])
            y_comp = lab_src.loc[complete, 'label'].values.astype(int)
            result[f'complete_{split}'] = (
                torch.FloatTensor(dfs[1].loc[complete].values),
                torch.FloatTensor(dfs[2].loc[complete].values),
                torch.FloatTensor(dfs[3].loc[complete].values),
                y_comp,
                complete,
            )
        else:
            result[f'complete_{split}'] = (
                torch.zeros(0), torch.zeros(0), torch.zeros(0),
                np.array([]), [],
            )

    result['n_mirna_features'] = merged['mirna'][0].shape[1]
    result['n_rnaseq_features'] = merged['rnaseq'][0].shape[1]
    result['n_meth_features'] = merged['meth'][0].shape[1]
    return result


def _print_and_save_cv(fold_results, fold_roc_data, fold_losses, fold_accuracies, args):
    """Print and save cross-validation summary."""
    print(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY\n{'='*70}")
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
        f.write(f"KD-SVAE-VCDN {args.n_folds}-Fold Cross-Validation Results\n")
        distill_str = "DISABLED" if args.no_distillation else "ENABLED"
        f.write(f"Knowledge Distillation: {distill_str}\n")
        f.write("(Metrics with optimal threshold per fold)\n")
        f.write("=" * 50 + "\n\n")
        for m in metrics:
            f.write(f"{m:20s}: {cv[f'{m}_mean']:.4f} +/- {cv[f'{m}_std']:.4f}\n")
        f.write("\n\nPer-Fold Results:\n" + "-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"\nFold {i+1} (threshold = {r['optimal_threshold']:.3f}):\n")
            for m in metrics:
                f.write(f"  {m}: {r[m]:.4f}\n")
    print(f"\nResults saved to: {path}")

    plot_roc_curves_cv(fold_roc_data,
                       save_path=os.path.join(args.save_dir, 'roc_curves_cv.png'),
                       title=f"ROC Curves – {args.n_folds}-Fold CV")
    plot_loss_with_ci(fold_losses,
                      save_path=os.path.join(args.save_dir, 'student_loss_cv.png'),
                      title=f"Student Training Loss – {args.n_folds}-Fold CV")
    plot_accuracy_with_ci(fold_accuracies,
                          save_path=os.path.join(args.save_dir, 'student_accuracy_cv.png'),
                          title=f"Student Training Accuracy – {args.n_folds}-Fold CV")


# =========================================================================
# MAIN
# =========================================================================

def main():
    args = parse_args()

    print(f"\n{'='*70}\nKD-SVAE-VCDN TRAINING PIPELINE\n{'='*70}")
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Knowledge Distillation: {'DISABLED' if args.no_distillation else 'ENABLED'}")
    print(f"Student fusion: {args.fusion}")
    if args.regularization != 'none':
        reg_info = f"{args.regularization} (l1={args.reg_lambda_l1}, l2={args.reg_lambda_l2})"
        print(f"Regularization: {reg_info}")
    if args.cross_validation:
        print(f"Mode: {args.n_folds}-Fold Cross-Validation")
    if args.early_stop_acc:
        print(f"Early stopping at accuracy: {args.early_stop_acc}")
    if args.kl_annealing != 'none':
        print(f"KL Annealing: {args.kl_annealing} (beta_max={args.kl_beta_max}, "
              f"warmup={args.kl_warmup_epochs}, cycle_length={args.kl_cycle_length})")
    print(f"Architecture (Level 1): {layer_size_te1}, latent={latent_dim_te1}")
    print(f"Architecture (Level 2): {layer_size_te2}, latent={latent_dim_te2}")
    print(f"Architecture (Student): {stu_layers_size}, latent={stu_latent_dim}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data with correct per-modality subsets
    preprocessed_dir = os.path.join(os.path.dirname(__file__), args.data_dir, 'preprocessed')
    kd_data = prepare_kd_data(preprocessed_dir)

    print(f"\nFeature dimensions:")
    print(f"  miRNA: {kd_data['n_mirna_features']}")
    print(f"  RNAseq: {kd_data['n_rnaseq_features']}")
    print(f"  Methylation: {kd_data['n_meth_features']}")

    if args.cross_validation:
        results = run_cross_validation(args, kd_data)
    else:
        results = train_fold(kd_data, args, fold_idx=None)

        print(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")
        print(f"Final Test Results (Threshold = {results['optimal_threshold']:.3f}):")
        for m in ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc'):
            print(f"  {m}: {results[m]:.4f}")
        print(f"\nModels saved to: {args.save_dir}")

        rpath = os.path.join(args.save_dir, 'results.txt')
        with open(rpath, 'w') as f:
            f.write("KD-SVAE-VCDN Results\n" + "=" * 40 + "\n")
            distill_str = "DISABLED" if args.no_distillation else "ENABLED"
            f.write(f"Knowledge Distillation: {distill_str}\n")
            f.write(f"Optimal Threshold: {results['optimal_threshold']:.4f}\n")
            for m in ('accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc'):
                f.write(f"{m}: {results[m]:.4f}\n")

        plot_roc_curve_single(
            results['y_test'], results['y_proba'], results['auc'],
            save_path=os.path.join(args.save_dir, 'roc_curve.png'),
            title="ROC Curve – KD-SVAE-VCDN")
        plot_training_curves(
            results['loss_history'], results['acc_history'],
            save_path=os.path.join(args.save_dir, 'training_curves.png'),
            title="Student Training Curves")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == '__main__':
    main()
