# -*- coding: utf-8 -*-
"""
Training and Testing module for the Top-Down KD-SVAE-VCDN architecture.

Provides training loops for each level of the top-down hierarchy:
  L1 all-modality teacher  →  L2 pairwise teachers  →  L3 students  →  integration

Reuses utilities (compute_beta, to_categorical, plotting, …) from train_test.py.

Author: Alberto Bastero
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from config import device, num_class as default_num_class
from KD import Teacher
from KD_topdown import (
    TopDownTeacherL1, IntegrationModule,
    loss_topdown_l1, loss_topdown_l2, loss_topdown_l3, loss_integration,
)
from train_test import (
    compute_beta, compute_l1_penalty, to_categorical, find_optimal_threshold,
)


###########################################################################
# LEVEL 1 — ALL-MODALITY TEACHER
###########################################################################

def training_topdown_l1(data1, data2, data3, num_class, layers_size, lr,
                        latent_dim, y, y_cat, num_epoch, batch_size,
                        temperature, save_dir='./checkpoints',
                        class_weights=None, gamma=2.0, early_stop_acc=None,
                        kl_annealing='none', kl_beta_max=0.1,
                        kl_warmup_epochs=None, kl_cycle_length=None,
                        fusion_mode='vcdn',
                        regularization='none', reg_lambda_l1=1e-5,
                        reg_lambda_l2=1e-4):
    """Train the top-down L1 all-modality teacher (no KD)."""

    d1 = (data1 if torch.is_tensor(data1)
          else torch.from_numpy(data1)).float().to(device)
    d2 = (data2 if torch.is_tensor(data2)
          else torch.from_numpy(data2)).float().to(device)
    d3 = (data3 if torch.is_tensor(data3)
          else torch.from_numpy(data3)).float().to(device)
    y_cat_t = (y_cat if torch.is_tensor(y_cat)
               else torch.from_numpy(y_cat)).float().to(device)

    y_np = y if isinstance(y, np.ndarray) else np.array(y)
    n_samples = len(d1)
    num_of_batch = (n_samples + batch_size - 1) // batch_size

    model = TopDownTeacherL1(data1, data2, data3, num_class, layers_size,
                             latent_dim, temperature, fusion_mode=fusion_mode)
    model = model.to(device)

    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    total_loss, total_acc = [], []

    print("Start Top-Down L1 Teacher training:")

    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing,
                            warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print(f"\nEpoch {epoch} (beta={beta:.4f})")

        epoch_pred, epoch_true, epoch_loss = [], [], []
        softs = np.zeros((n_samples, num_class))
        model.train()

        perm = torch.randperm(n_samples)
        d1s, d2s, d3s = d1[perm], d2[perm], d3[perm]
        ycs = y_cat_t[perm]
        y_shuf = y_np[perm.cpu().numpy()]

        for step in range(num_of_batch):
            s, e = step * batch_size, min((step + 1) * batch_size, n_samples)
            b1, b2, b3, yb = d1s[s:e], d2s[s:e], d3s[s:e], ycs[s:e]
            if len(b1) == 0:
                continue

            optimizer.zero_grad()
            (r1, r2, r3, m1, m2, m3, lv1, lv2, lv3,
             pred, soft) = model(b1, b2, b3)

            loss_val, _ = loss_topdown_l1(
                b1, b2, b3, r1, r2, r3, m1, m2, m3, lv1, lv2, lv3,
                pred, yb, class_weights=class_weights, gamma=gamma, beta=beta,
            )
            if use_l1:
                loss_val = loss_val + reg_lambda_l1 * compute_l1_penalty(model)
            loss_val.backward()
            optimizer.step()

            epoch_loss.append(loss_val.item())
            epoch_pred.append(torch.argmax(pred, dim=-1).detach().cpu().numpy())
            epoch_true.append(y_shuf[s:e])

            orig_idx = perm[s:e].cpu().numpy()
            softs[orig_idx] = soft.detach().cpu().numpy()

        acc = accuracy_score(np.concatenate(epoch_true),
                             np.concatenate(epoch_pred))
        total_loss.append(np.mean(epoch_loss))
        total_acc.append(acc)
        print(f"  loss={total_loss[-1]:.4f}  acc={acc:.4f}")

        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping (acc {acc:.4f} >= {early_stop_acc})")
            break

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(save_dir, 'topdown_l1_teacher.pt'))

    return torch.from_numpy(softs).float().to(device)


def get_topdown_l1_predictions(data1, data2, data3, model_path, num_class,
                               layers_size, latent_dim, temperature,
                               fusion_mode='vcdn'):
    """Generate soft labels from a trained top-down L1 teacher (eval mode)."""
    d1 = (data1 if torch.is_tensor(data1)
          else torch.from_numpy(data1)).float().to(device)
    d2 = (data2 if torch.is_tensor(data2)
          else torch.from_numpy(data2)).float().to(device)
    d3 = (data3 if torch.is_tensor(data3)
          else torch.from_numpy(data3)).float().to(device)

    model = TopDownTeacherL1(data1, data2, data3, num_class, layers_size,
                             latent_dim, temperature, fusion_mode=fusion_mode)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        *_, soft_labels = model(d1, d2, d3)
    return soft_labels


###########################################################################
# LEVEL 2 — PAIRWISE TEACHERS (KD from 1 L1 teacher)
###########################################################################

def training_topdown_l2(data, num_class, layers_size_te1, layers_size_te2, lr,
                        latent_dim_te1, latent_dim_te2,
                        y, y_cat, soft_labels_l1, kd_weight,
                        num_epoch, batch_size, temperature,
                        pair_tag='12', save_dir='./checkpoints',
                        class_weights=None, gamma=2.0, early_stop_acc=None,
                        kl_annealing='none', kl_beta_max=0.1,
                        kl_warmup_epochs=None, kl_cycle_length=None,
                        regularization='none', reg_lambda_l1=1e-5,
                        reg_lambda_l2=1e-4):
    """Train a top-down L2 pairwise teacher with KD from the L1 teacher."""

    data_t = (data if torch.is_tensor(data)
              else torch.from_numpy(data)).float().to(device)
    y_cat_t = (y_cat if torch.is_tensor(y_cat)
               else torch.from_numpy(y_cat)).float().to(device)
    if isinstance(soft_labels_l1, np.ndarray):
        soft_labels_l1 = torch.from_numpy(soft_labels_l1).float().to(device)
    soft_labels_l1 = soft_labels_l1.to(device)

    y_np = y if isinstance(y, np.ndarray) else np.array(y)
    n_samples = len(data_t)
    num_of_batch = (n_samples + batch_size - 1) // batch_size

    model = Teacher(data, num_class, layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2, temperature, step=2)
    model = model.to(device)

    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    total_loss, total_acc = [], []
    distillation_losses = {'dl': []}

    print(f"Start Top-Down L2 Teacher training (pair {pair_tag}):")

    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing,
                            warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print(f"\nEpoch {epoch} (beta={beta:.4f})")

        epoch_pred, epoch_true, epoch_loss, epoch_dl = [], [], [], []
        softs = np.zeros((n_samples, num_class))
        model.train()

        perm = torch.randperm(n_samples)
        data_shuf = data_t[perm]
        yc_shuf = y_cat_t[perm]
        sl_shuf = soft_labels_l1[perm]
        y_shuf = y_np[perm.cpu().numpy()]

        for step in range(num_of_batch):
            s, e = step * batch_size, min((step + 1) * batch_size, n_samples)
            db, yb, slb = data_shuf[s:e], yc_shuf[s:e], sl_shuf[s:e]
            if len(db) == 0:
                continue

            optimizer.zero_grad()
            recon, means, log_var, pred, soft = model(db)

            loss_val, comps = loss_topdown_l2(
                db, recon, means, log_var, yb, pred,
                slb, kd_weight,
                class_weights=class_weights, gamma=gamma, beta=beta,
            )
            if use_l1:
                loss_val = loss_val + reg_lambda_l1 * compute_l1_penalty(model)
            loss_val.backward()
            optimizer.step()

            epoch_loss.append(loss_val.item())
            epoch_dl.append(comps['dl'])
            epoch_pred.append(torch.argmax(pred, -1).detach().cpu().numpy())
            epoch_true.append(y_shuf[s:e])

            orig_idx = perm[s:e].cpu().numpy()
            softs[orig_idx] = soft.detach().cpu().numpy()

        acc = accuracy_score(np.concatenate(epoch_true),
                             np.concatenate(epoch_pred))
        total_loss.append(np.mean(epoch_loss))
        total_acc.append(acc)
        distillation_losses['dl'].append(np.mean(epoch_dl))
        print(f"  loss={total_loss[-1]:.4f}  acc={acc:.4f}")

        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping (acc {acc:.4f} >= {early_stop_acc})")
            break

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(save_dir, f'topdown_l2_te_{pair_tag}.pt'))

    return torch.from_numpy(softs).float().to(device), distillation_losses


###########################################################################
# LEVEL 3 — SINGLE-MODALITY STUDENTS (KD from 2 L2 teachers)
###########################################################################

def training_topdown_l3(data, num_class, layers_size_te1, layers_size_te2, lr,
                        latent_dim_te1, latent_dim_te2,
                        y, y_cat, soft_labels_l2a, soft_labels_l2b, a, b,
                        num_epoch, batch_size, temperature,
                        mod_id=1, save_dir='./checkpoints',
                        class_weights=None, gamma=2.0, early_stop_acc=None,
                        kl_annealing='none', kl_beta_max=0.1,
                        kl_warmup_epochs=None, kl_cycle_length=None,
                        regularization='none', reg_lambda_l1=1e-5,
                        reg_lambda_l2=1e-4):
    """Train a top-down L3 single-modality student with KD from 2 L2 teachers."""

    data_t = (data if torch.is_tensor(data)
              else torch.from_numpy(data)).float().to(device)
    y_cat_t = (y_cat if torch.is_tensor(y_cat)
               else torch.from_numpy(y_cat)).float().to(device)
    if isinstance(soft_labels_l2a, np.ndarray):
        soft_labels_l2a = torch.from_numpy(soft_labels_l2a).float().to(device)
    if isinstance(soft_labels_l2b, np.ndarray):
        soft_labels_l2b = torch.from_numpy(soft_labels_l2b).float().to(device)
    soft_labels_l2a = soft_labels_l2a.to(device)
    soft_labels_l2b = soft_labels_l2b.to(device)

    y_np = y if isinstance(y, np.ndarray) else np.array(y)
    n_samples = len(data_t)
    num_of_batch = (n_samples + batch_size - 1) // batch_size

    model = Teacher(data, num_class, layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2, temperature, step=1)
    model = model.to(device)

    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    total_loss, total_acc = [], []
    distillation_losses = {'dl1': [], 'dl2': []}

    mod_names = {1: 'miRNA', 2: 'RNAseq', 3: 'Methylation'}
    print(f"Start Top-Down L3 Student training ({mod_names.get(mod_id, mod_id)}):")

    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing,
                            warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print(f"\nEpoch {epoch} (beta={beta:.4f})")

        ep_pred, ep_true, ep_loss = [], [], []
        ep_dl1, ep_dl2 = [], []
        model.train()

        perm = torch.randperm(n_samples)
        data_shuf = data_t[perm]
        yc_shuf = y_cat_t[perm]
        sla_shuf = soft_labels_l2a[perm]
        slb_shuf = soft_labels_l2b[perm]
        y_shuf = y_np[perm.cpu().numpy()]

        for step in range(num_of_batch):
            s, e = step * batch_size, min((step + 1) * batch_size, n_samples)
            db, yb = data_shuf[s:e], yc_shuf[s:e]
            sla_b, slb_b = sla_shuf[s:e], slb_shuf[s:e]
            if len(db) == 0:
                continue

            optimizer.zero_grad()
            recon, means, log_var, pred, _ = model(db)

            loss_val, comps = loss_topdown_l3(
                db, recon, means, log_var, yb, pred,
                sla_b, slb_b, a, b,
                class_weights=class_weights, gamma=gamma, beta=beta,
            )
            if use_l1:
                loss_val = loss_val + reg_lambda_l1 * compute_l1_penalty(model)
            loss_val.backward()
            optimizer.step()

            ep_loss.append(loss_val.item())
            ep_dl1.append(comps['dl1'])
            ep_dl2.append(comps['dl2'])
            ep_pred.append(torch.argmax(pred, -1).detach().cpu().numpy())
            ep_true.append(y_shuf[s:e])

        acc = accuracy_score(np.concatenate(ep_true), np.concatenate(ep_pred))
        total_loss.append(np.mean(ep_loss))
        total_acc.append(acc)
        distillation_losses['dl1'].append(np.mean(ep_dl1))
        distillation_losses['dl2'].append(np.mean(ep_dl2))
        print(f"  loss={total_loss[-1]:.4f}  acc={acc:.4f}")

        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping (acc {acc:.4f} >= {early_stop_acc})")
            break

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(save_dir, f'topdown_l3_stu_{mod_id}.pt'))

    return distillation_losses


###########################################################################
# INTEGRATION MODULE TRAINING
###########################################################################

def training_integration(data1, data2, data3, num_class,
                         layers_size_te1, layers_size_te2,
                         latent_dim_te1, latent_dim_te2,
                         integration_layers_size,
                         y, y_cat, lr, num_epoch, batch_size, temperature,
                         save_dir='./checkpoints',
                         class_weights=None, gamma=2.0,
                         fusion_mode='vcdn',
                         regularization='none', reg_lambda_l1=1e-5,
                         reg_lambda_l2=1e-4):
    """Train the integration module on frozen L3 student representations."""

    d1 = (data1 if torch.is_tensor(data1)
          else torch.from_numpy(data1)).float().to(device)
    d2 = (data2 if torch.is_tensor(data2)
          else torch.from_numpy(data2)).float().to(device)
    d3 = (data3 if torch.is_tensor(data3)
          else torch.from_numpy(data3)).float().to(device)
    y_cat_t = (y_cat if torch.is_tensor(y_cat)
               else torch.from_numpy(y_cat)).float().to(device)
    y_np = y if isinstance(y, np.ndarray) else np.array(y)

    n_samples = len(d1)
    num_of_batch = (n_samples + batch_size - 1) // batch_size

    # Load frozen L3 students
    students = []
    modality_data = [data1, data2, data3]
    for mod_id in [1, 2, 3]:
        stu = Teacher(modality_data[mod_id - 1], num_class,
                      layers_size_te1, layers_size_te2,
                      latent_dim_te1, latent_dim_te2, temperature, step=1)
        stu = stu.to(device)
        stu.load_state_dict(torch.load(
            os.path.join(save_dir, f'topdown_l3_stu_{mod_id}.pt'),
            map_location=device))
        stu.eval()
        for p in stu.parameters():
            p.requires_grad = False
        students.append(stu)

    integration = IntegrationModule(num_class, latent_dim_te1,
                                    integration_layers_size,
                                    fusion_mode=fusion_mode)
    integration = integration.to(device)

    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(integration.parameters(), lr=lr,
                                 weight_decay=wd)

    total_loss, total_acc = [], []

    print(f"Start Integration Module training (fusion={fusion_mode}):")

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch}")

        ep_pred, ep_true, ep_loss = [], [], []
        integration.train()

        perm = torch.randperm(n_samples)
        d1s, d2s, d3s = d1[perm], d2[perm], d3[perm]
        ycs = y_cat_t[perm]
        y_shuf = y_np[perm.cpu().numpy()]

        for step in range(num_of_batch):
            s, e = step * batch_size, min((step + 1) * batch_size, n_samples)
            b1, b2, b3, yb = d1s[s:e], d2s[s:e], d3s[s:e], ycs[s:e]
            if len(b1) == 0:
                continue

            with torch.no_grad():
                _, m1, lv1, _, _ = students[0](b1)
                _, m2, lv2, _, _ = students[1](b2)
                _, m3, lv3, _, _ = students[2](b3)

            optimizer.zero_grad()
            pred = integration([m1, m2, m3], [lv1, lv2, lv3])
            loss_val = loss_integration(pred, yb,
                                        class_weights=class_weights,
                                        gamma=gamma)
            loss_val.backward()
            optimizer.step()

            ep_loss.append(loss_val.item())
            ep_pred.append(torch.argmax(pred, -1).detach().cpu().numpy())
            ep_true.append(y_shuf[s:e])

        acc = accuracy_score(np.concatenate(ep_true), np.concatenate(ep_pred))
        total_loss.append(np.mean(ep_loss))
        total_acc.append(acc)
        print(f"  loss={total_loss[-1]:.4f}  acc={acc:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(integration.state_dict(),
               os.path.join(save_dir, 'topdown_integration.pt'))

    return total_loss, total_acc


###########################################################################
# TESTING
###########################################################################

def testing_topdown(x1_test, x2_test, x3_test, num_class,
                    layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2,
                    integration_layers_size,
                    y_test, y_test_cat, batch_size, temperature,
                    save_dir='./checkpoints',
                    optimize_threshold=True, fusion_mode='vcdn'):
    """Test the top-down pipeline (3 frozen students + integration module)."""

    x1 = (x1_test if torch.is_tensor(x1_test)
          else torch.from_numpy(x1_test)).float().to(device)
    x2 = (x2_test if torch.is_tensor(x2_test)
          else torch.from_numpy(x2_test)).float().to(device)
    x3 = (x3_test if torch.is_tensor(x3_test)
          else torch.from_numpy(x3_test)).float().to(device)

    # Load L3 students
    students = []
    test_data = [x1_test, x2_test, x3_test]
    for mod_id in [1, 2, 3]:
        stu = Teacher(test_data[mod_id - 1], num_class,
                      layers_size_te1, layers_size_te2,
                      latent_dim_te1, latent_dim_te2, temperature, step=1)
        stu = stu.to(device)
        stu.load_state_dict(torch.load(
            os.path.join(save_dir, f'topdown_l3_stu_{mod_id}.pt'),
            map_location=device))
        stu.eval()
        students.append(stu)

    # Load integration module
    integration = IntegrationModule(num_class, latent_dim_te1,
                                    integration_layers_size,
                                    fusion_mode=fusion_mode)
    integration = integration.to(device)
    integration.load_state_dict(torch.load(
        os.path.join(save_dir, 'topdown_integration.pt'),
        map_location=device))
    integration.eval()

    with torch.no_grad():
        _, m1, lv1, _, _ = students[0](x1)
        _, m2, lv2, _, _ = students[1](x2)
        _, m3, lv3, _, _ = students[2](x3)
        pred_labels_score = integration([m1, m2, m3], [lv1, lv2, lv3])

    pred_np = pred_labels_score.cpu().numpy()
    y_proba = pred_np[:, 1]
    test_pred_default = np.argmax(pred_np, axis=1)

    auc = roc_auc_score(y_test, y_proba)
    acc_def = accuracy_score(y_test, test_pred_default)
    bal_acc_def = balanced_accuracy_score(y_test, test_pred_default)
    f1_def = f1_score(y_test, test_pred_default)
    prec_def = precision_score(y_test, test_pred_default, zero_division=0)
    rec_def = recall_score(y_test, test_pred_default, zero_division=0)

    print("Test Results (Default Threshold = 0.5):")
    print(f"  Accuracy: {acc_def:.4f}")
    print(f"  Balanced Accuracy: {bal_acc_def:.4f}")
    print(f"  F1 Score: {f1_def:.4f}")
    print(f"  Precision: {prec_def:.4f}")
    print(f"  Recall: {rec_def:.4f}")
    print(f"  AUC: {auc:.4f}")

    optimal_threshold = 0.5
    if optimize_threshold:
        optimal_threshold = find_optimal_threshold(y_test, y_proba,
                                                   method='balanced')
        test_pred_opt = (y_proba >= optimal_threshold).astype(int)

        acc_opt = accuracy_score(y_test, test_pred_opt)
        bal_acc_opt = balanced_accuracy_score(y_test, test_pred_opt)
        f1_opt = f1_score(y_test, test_pred_opt)
        prec_opt = precision_score(y_test, test_pred_opt, zero_division=0)
        rec_opt = recall_score(y_test, test_pred_opt, zero_division=0)

        print(f"\nTest Results (Optimal Threshold = {optimal_threshold:.3f}):")
        print(f"  Accuracy: {acc_opt:.4f}")
        print(f"  Balanced Accuracy: {bal_acc_opt:.4f}")
        print(f"  F1 Score: {f1_opt:.4f}")
        print(f"  Precision: {prec_opt:.4f}")
        print(f"  Recall: {rec_opt:.4f}")
        print(f"  AUC: {auc:.4f}")

        return (f1_opt, auc, y_proba, bal_acc_opt,
                prec_opt, rec_opt, acc_opt, optimal_threshold)

    return (f1_def, auc, y_proba, bal_acc_def,
            prec_def, rec_def, acc_def, optimal_threshold)
