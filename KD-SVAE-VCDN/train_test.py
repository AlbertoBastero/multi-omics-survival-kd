# -*- coding: utf-8 -*-
"""
Training and Testing module for KD-SVAE-VCDN.
Translated to PyTorch from the original Keras implementation.

Author: Alberto Bastero (translated)
Original author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

from KD import *
from config import *
# from preprocess import *  # Uncomment when preprocess module is available


def compute_beta(epoch, num_epochs, beta_max=0.1, annealing='none',
                 warmup_epochs=None, cycle_length=None):
    """Compute the KL weight (beta) for the current epoch.

    Args:
        epoch: Current epoch (0-indexed).
        num_epochs: Total number of training epochs.
        beta_max: Target maximum beta value.
        annealing: 'none' (constant), 'linear', or 'cyclical'.
        warmup_epochs: Number of epochs for linear warmup (default: num_epochs // 2).
        cycle_length: Length of one cycle for cyclical annealing (default: num_epochs // 4).

    Returns:
        Beta value for this epoch.
    """
    if annealing == 'none':
        return beta_max

    if annealing == 'linear':
        if warmup_epochs is None:
            warmup_epochs = max(num_epochs // 2, 1)
        return min(beta_max, epoch / warmup_epochs * beta_max)

    if annealing == 'cyclical':
        if cycle_length is None:
            cycle_length = max(num_epochs // 4, 1)
        # Within each cycle, linearly ramp up over the first half
        position = epoch % cycle_length
        ratio = 0.5  # fraction of cycle spent ramping
        return min(beta_max, position / (cycle_length * ratio) * beta_max)

    raise ValueError(f"Unknown annealing strategy: {annealing}")


def compute_l1_penalty(model):
    """Compute the L1 norm of all model parameters (sum of absolute values)."""
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1 = l1 + param.abs().sum()
    return l1


###########################################################################
# UTILITY FUNCTIONS
###########################################################################

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    """
    Calculate sample weights using inverse class frequency (sklearn convention).
    
    Weight for class i = n_samples / (n_classes * n_samples_of_class_i)
    Minority classes get higher weights, majority classes get lower weights.
    
    Args:
        labels: Array of class labels
        num_class: Number of classes
        use_sample_weight: Whether to use sample weighting
        
    Returns:
        Array of sample weights
    """
    if not use_sample_weight:
        return np.ones(len(labels))
    
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = len(labels) / (num_class * count[i])
           
    return sample_weight


def to_categorical(labels, num_classes):
    """
    Convert integer labels to one-hot encoded format.
    PyTorch equivalent of keras.utils.to_categorical()
    
    Args:
        labels: Integer labels (numpy array or torch tensor)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded tensor
    """
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    return F.one_hot(labels, num_classes).float()


###########################################################################
# TEACHER TRAINING FUNCTIONS
###########################################################################

def training_te_level1(data, num_class, layers_size_te1, layers_size_te2, lr,
                       latent_dim_te1, latent_dim_te2, y, y_cat, num_epoch,
                       batch_size, temperature,
                       steps=1, num_te=1, save_dir='./checkpoints',
                       class_weights=None, gamma=2.0, early_stop_acc=None,
                       use_sample_weight=None,
                       kl_annealing='none', kl_beta_max=0.1,
                       kl_warmup_epochs=None, kl_cycle_length=None,
                       regularization='none', reg_lambda_l1=1e-5, reg_lambda_l2=1e-4):
    """
    Train a teacher model at level 1 (no knowledge distillation).
    
    Key PyTorch translations from Keras:
    - tf.GradientTape() → loss.backward() + optimizer.step()
    - optimizer.apply_gradients() → optimizer.step()
    - model.trainable_weights → model.parameters()
    
    Args:
        data: Input data (numpy array or tensor)
        num_class: Number of output classes
        layers_size_te1, layers_size_te2: Layer sizes for teacher architectures
        lr: Learning rate
        latent_dim_te1, latent_dim_te2: Latent dimensions
        y: Integer labels
        y_cat: One-hot encoded labels
        num_epoch: Number of training epochs
        batch_size: Batch size
        temperature: Temperature for soft labels
        use_sample_weight: Whether to use sample weighting
        steps: Teacher step (1 or 2)
        num_te: Teacher number (for saving)
        save_dir: Directory to save model weights
        
    Returns:
        Soft labels tensor from the trained model
    """
    # Convert data to tensors if needed
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(device)
    else:
        data_tensor = data.float().to(device)
    
    if isinstance(y_cat, np.ndarray):
        y_cat_tensor = torch.from_numpy(y_cat).float().to(device)
    else:
        y_cat_tensor = y_cat.float().to(device)
    
    n_samples = len(data_tensor)
    num_of_batch = (n_samples + batch_size - 1) // batch_size
    
    model = Teacher(data, num_class, layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2, temperature, step=steps)
    model = model.to(device)
    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    # Keep y as numpy for metric computation
    y_np = y if isinstance(y, np.ndarray) else y.copy()

    total_loss = []
    total_acc = []

    print(f"Start Teacher training in step ({steps}):")
    
    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing, warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print("Teacher training")
        print(f"\nStart of epoch {epoch} (beta={beta:.4f})")

        epoch_pred = []
        epoch_true = []
        epoch_loss = []
        preds = np.zeros(shape=(n_samples, num_class))
        softs = np.zeros(shape=(n_samples, num_class))

        model.train()

        # Shuffle data at each epoch
        perm = torch.randperm(n_samples)
        data_shuf = data_tensor[perm]
        y_cat_shuf = y_cat_tensor[perm]
        y_shuf = y_np[perm.cpu().numpy()]

        for step in range(num_of_batch):
            start_idx = step * batch_size
            end_idx = min(batch_size * (step + 1), n_samples)

            data_batch = data_shuf[start_idx:end_idx]
            y_batch = y_cat_shuf[start_idx:end_idx]

            if len(data_batch) == 0:
                continue

            optimizer.zero_grad()
            recon, means, log_var, pred_labels, soft_labels = model(data_batch)
            loss_value = loss_teacher_level1(data_batch, recon, means, log_var,
                                             y_batch, pred_labels,
                                             class_weights=class_weights, gamma=gamma,
                                             beta=beta)
            if use_l1:
                loss_value = loss_value + reg_lambda_l1 * compute_l1_penalty(model)
            loss_value.backward()
            optimizer.step()

            epoch_loss.append(loss_value.item())
            
            y_pred = torch.argmax(pred_labels, dim=-1).detach().cpu().numpy()
            epoch_pred.append(y_pred)
            epoch_true.append(y_shuf[start_idx:end_idx])
            
            # Store predictions/softs in original order
            orig_idx = perm[start_idx:end_idx].cpu().numpy()
            preds[orig_idx, :] = pred_labels.detach().cpu().numpy()
            softs[orig_idx, :] = soft_labels.detach().cpu().numpy()
        
        prediction = np.concatenate(epoch_pred)
        true_labels = np.concatenate(epoch_true)
        mean_loss_train = sum(epoch_loss) / len(epoch_loss)
        acc = accuracy_score(true_labels, prediction)
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)
        print(f"epoch: {epoch} train_loss: {mean_loss_train:.4f}, acc: {acc:.4f}")
        
        # Early stopping based on accuracy
        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping triggered! Accuracy {acc:.4f} >= target {early_stop_acc:.4f}")
            break
    
    # Save model weights
    # PyTorch: torch.save(model.state_dict(), path) replaces model.save_weights()
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/KD_TE_{steps}{num_te}.pt')
    
    # Convert to tensor and return
    softs_tensor = torch.from_numpy(softs).float().to(device)
    return softs_tensor


def training_te_level2(data, num_class, layers_size_te1, layers_size_te2, lr,
                       latent_dim_te1, latent_dim_te2, y, y_cat,
                       softs1, softs2, a, b, num_epoch, batch_size, temperature,
                       steps=2, num_te=1,
                       save_dir='./checkpoints', class_weights=None, gamma=2.0,
                       early_stop_acc=None, use_sample_weight=None,
                       kl_annealing='none', kl_beta_max=0.1,
                       kl_warmup_epochs=None, kl_cycle_length=None,
                       regularization='none', reg_lambda_l1=1e-5, reg_lambda_l2=1e-4):
    """
    Train a teacher model at level 2 (with knowledge distillation from other teachers).
    
    Args:
        data: Input data
        softs1, softs2: Soft labels from level 1 teachers for distillation
        a, b: Weights for distillation losses
        (other args same as training_te_level1)
        
    Returns:
        Soft labels tensor from the trained model
    """
    # Convert to tensors
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(device)
    else:
        data_tensor = data.float().to(device)
    
    if isinstance(y_cat, np.ndarray):
        y_cat_tensor = torch.from_numpy(y_cat).float().to(device)
    else:
        y_cat_tensor = y_cat.float().to(device)
    
    if isinstance(softs1, np.ndarray):
        softs1 = torch.from_numpy(softs1).float().to(device)
    if isinstance(softs2, np.ndarray):
        softs2 = torch.from_numpy(softs2).float().to(device)
    
    n_samples = len(data_tensor)
    num_of_batch = (n_samples + batch_size - 1) // batch_size
    
    model = Teacher(data, num_class, layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2, temperature, step=steps)
    model = model.to(device)
    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    y_np = y if isinstance(y, np.ndarray) else y.copy()

    total_loss = []
    total_acc = []
    distillation_losses = {'dl1': [], 'dl2': []}

    print(f"Start Teacher training in step ({steps}):")

    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing, warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print("Teacher training")
        print(f"\nStart of epoch {epoch} (beta={beta:.4f})")

        epoch_pred = []
        epoch_true = []
        epoch_loss = []
        epoch_dl1 = []
        epoch_dl2 = []
        preds = np.zeros(shape=(n_samples, num_class))
        softs = np.zeros(shape=(n_samples, num_class))

        model.train()

        # Shuffle data at each epoch
        perm = torch.randperm(n_samples)
        data_shuf = data_tensor[perm]
        y_cat_shuf = y_cat_tensor[perm]
        y_shuf = y_np[perm.cpu().numpy()]
        softs1_shuf = softs1[perm]
        softs2_shuf = softs2[perm]

        for step in range(num_of_batch):
            start_idx = step * batch_size
            end_idx = min(batch_size * (step + 1), n_samples)

            data_batch = data_shuf[start_idx:end_idx]
            y_batch = y_cat_shuf[start_idx:end_idx]
            softs1_batch = softs1_shuf[start_idx:end_idx]
            softs2_batch = softs2_shuf[start_idx:end_idx]

            if len(data_batch) == 0:
                continue

            optimizer.zero_grad()
            recon, means, log_var, pred_labels, soft_labels = model(data_batch)
            loss_value, loss_components = loss_teacher_level2(data_batch, recon, means, log_var,
                                             y_batch, pred_labels,
                                             softs1_batch, softs2_batch,
                                             a, b,
                                             class_weights=class_weights, gamma=gamma,
                                             beta=beta)
            if use_l1:
                loss_value = loss_value + reg_lambda_l1 * compute_l1_penalty(model)
            loss_value.backward()
            optimizer.step()
            
            epoch_loss.append(loss_value.item())
            epoch_dl1.append(loss_components['dl1'])
            epoch_dl2.append(loss_components['dl2'])
            
            y_pred = torch.argmax(pred_labels, dim=-1).detach().cpu().numpy()
            epoch_pred.append(y_pred)
            epoch_true.append(y_shuf[start_idx:end_idx])
            
            orig_idx = perm[start_idx:end_idx].cpu().numpy()
            preds[orig_idx, :] = pred_labels.detach().cpu().numpy()
            softs[orig_idx, :] = soft_labels.detach().cpu().numpy()
        
        prediction = np.concatenate(epoch_pred)
        true_labels = np.concatenate(epoch_true)
        mean_loss_train = sum(epoch_loss) / len(epoch_loss)
        acc = accuracy_score(true_labels, prediction)
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)
        print(f"epoch: {epoch} train_loss: {mean_loss_train:.4f}, acc: {acc:.4f}")
        
        # Store average distillation losses for this epoch
        if len(epoch_dl1) > 0:
            distillation_losses['dl1'].append(np.mean(epoch_dl1))
        if len(epoch_dl2) > 0:
            distillation_losses['dl2'].append(np.mean(epoch_dl2))
        
        # Early stopping based on accuracy
        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping triggered! Accuracy {acc:.4f} >= target {early_stop_acc:.4f}")
            break
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/KD_TE_{steps}{num_te}.pt')
    
    softs_tensor = torch.from_numpy(softs).float().to(device)
    return softs_tensor, distillation_losses  # Return distillation losses


def get_predictions(x, model_path, num_class, layers_size_te1, layers_size_te2,
                    latent_dim_te1, latent_dim_te2, temperature=1.5, step=1):
    """
    Get soft label predictions from a trained teacher model.
    
    Key PyTorch translations:
    - tf.convert_to_tensor() → torch.from_numpy() or torch.tensor()
    - model.load_weights() → model.load_state_dict(torch.load())
    
    Args:
        x: Input data
        model_path: Path to saved model weights
        step: Teacher step (1 or 2)
        
    Returns:
        Soft labels tensor
    """
    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x).float().to(device)
    else:
        x_tensor = x.float().to(device)
    
    # Create model and load weights
    teacher = Teacher(x, num_class, layers_size_te1, layers_size_te2,
                      latent_dim_te1, latent_dim_te2, temperature, step=step)
    teacher = teacher.to(device)
    
    # PyTorch: load_state_dict(torch.load()) replaces load_weights()
    teacher.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode (disables dropout, etc.)
    teacher.eval()
    
    # No gradient computation needed for inference
    with torch.no_grad():
        _, _, _, _, soft_labels = teacher(x_tensor)
    
    return soft_labels


###########################################################################
# STUDENT TRAINING FUNCTION
###########################################################################

def training_stu(data1, data2, data3, num_class, layers_size, lr, stu_latent_dim,
                 y, y_cat, softs1, softs2, softs3, a, b, c, num_epoch,
                 batch_size, temperature,
                 save_dir='./checkpoints', class_weights=None, gamma=2.0,
                 early_stop_acc=None, use_sample_weight=None,
                 kl_annealing='none', kl_beta_max=0.1,
                 kl_warmup_epochs=None, kl_cycle_length=None,
                 fusion_mode='vcdn',
                 regularization='none', reg_lambda_l1=1e-5, reg_lambda_l2=1e-4):
    """
    Train the student model with knowledge distillation from all teachers.
    
    Args:
        data1, data2, data3: Multi-omics data from each view
        softs1, softs2, softs3: Soft labels from teachers for distillation
        a, b, c: Weights for distillation losses
        (other args similar to teacher training)
        
    Returns:
        Tuple of (loss_history, accuracy_history, final_confusion_matrix)
    """
    # Convert to tensors
    if isinstance(data1, np.ndarray):
        data1_tensor = torch.from_numpy(data1).float().to(device)
    else:
        data1_tensor = data1.float().to(device)
    
    if isinstance(data2, np.ndarray):
        data2_tensor = torch.from_numpy(data2).float().to(device)
    else:
        data2_tensor = data2.float().to(device)
    
    if isinstance(data3, np.ndarray):
        data3_tensor = torch.from_numpy(data3).float().to(device)
    else:
        data3_tensor = data3.float().to(device)
    
    if isinstance(y_cat, np.ndarray):
        y_cat_tensor = torch.from_numpy(y_cat).float().to(device)
    else:
        y_cat_tensor = y_cat.float().to(device)
    
    if isinstance(softs1, np.ndarray):
        softs1 = torch.from_numpy(softs1).float().to(device)
    if isinstance(softs2, np.ndarray):
        softs2 = torch.from_numpy(softs2).float().to(device)
    if isinstance(softs3, np.ndarray):
        softs3 = torch.from_numpy(softs3).float().to(device)
    
    n_samples = len(data1_tensor)
    num_of_batch = (n_samples + batch_size - 1) // batch_size
    
    stu = Student(data1, data2, data3, num_class, layers_size, stu_latent_dim,
                  batch_size, fusion_mode=fusion_mode)
    stu = stu.to(device)
    wd = reg_lambda_l2 if regularization in ('l2', 'elastic') else 0.0
    optimizer = torch.optim.Adam(stu.parameters(), lr=lr, weight_decay=wd)
    use_l1 = regularization in ('l1', 'elastic')

    y_np = y if isinstance(y, np.ndarray) else y.copy()

    total_loss = []
    total_acc = []
    distillation_losses = {'dl1': [], 'dl2': [], 'dl3': []}
    component_histories = {'kld': [], 'bce': [], 'beta': []}
    conf = None

    print(f"Start STU training (fusion={fusion_mode})")
    
    for epoch in range(num_epoch):
        beta = compute_beta(epoch, num_epoch, beta_max=kl_beta_max,
                            annealing=kl_annealing, warmup_epochs=kl_warmup_epochs,
                            cycle_length=kl_cycle_length)
        print("STU training")
        print(f"\nStart of epoch {epoch} (beta={beta:.4f})")

        epoch_pred = []
        epoch_true = []
        epoch_loss = []
        epoch_dl1 = []
        epoch_dl2 = []
        epoch_dl3 = []
        epoch_kld = []
        epoch_bce = []

        stu.train()
        
        # Shuffle data at each epoch
        perm = torch.randperm(n_samples)
        d1_shuf = data1_tensor[perm]
        d2_shuf = data2_tensor[perm]
        d3_shuf = data3_tensor[perm]
        y_cat_shuf = y_cat_tensor[perm]
        y_shuf = y_np[perm.cpu().numpy()]
        s1_shuf = softs1[perm]
        s2_shuf = softs2[perm]
        s3_shuf = softs3[perm]
        
        for step in range(num_of_batch):
            start_idx = step * batch_size
            end_idx = min(batch_size * (step + 1), n_samples)
            
            data1_batch = d1_shuf[start_idx:end_idx]
            data2_batch = d2_shuf[start_idx:end_idx]
            data3_batch = d3_shuf[start_idx:end_idx]
            y_batch = y_cat_shuf[start_idx:end_idx]
            
            soft1_batch = s1_shuf[start_idx:end_idx]
            soft2_batch = s2_shuf[start_idx:end_idx]
            soft3_batch = s3_shuf[start_idx:end_idx]
            
            if len(data1_batch) == 0:
                continue
            
            optimizer.zero_grad()
            
            (recon_1, recon_2, recon_3,
             means_1, means_2, means_3,
             log_var_1, log_var_2, log_var_3,
             pred_labels) = stu(data1_batch, data2_batch, data3_batch)
            
            loss_value, loss_components = loss_student(
                data1_batch, data2_batch, data3_batch,
                recon_1, recon_2, recon_3,
                means_1, means_2, means_3,
                log_var_1, log_var_2, log_var_3,
                pred_labels, y_batch,
                soft1_batch, soft2_batch, soft3_batch,
                a, b, c,
                class_weights=class_weights, gamma=gamma,
                beta=beta
            )
            if use_l1:
                loss_value = loss_value + reg_lambda_l1 * compute_l1_penalty(stu)

            loss_value.backward()
            optimizer.step()
            
            epoch_loss.append(loss_value.item())
            epoch_dl1.append(loss_components['dl1'])
            epoch_dl2.append(loss_components['dl2'])
            epoch_dl3.append(loss_components['dl3'])
            epoch_kld.append(loss_components['kld'])
            epoch_bce.append(loss_components['bce'])
            
            y_pred = torch.argmax(pred_labels, dim=-1).detach().cpu().numpy()
            epoch_pred.append(y_pred)
            epoch_true.append(y_shuf[start_idx:end_idx])
        
        prediction = np.concatenate(epoch_pred)
        true_labels = np.concatenate(epoch_true)
        mean_loss_train = sum(epoch_loss) / len(epoch_loss)
        acc = accuracy_score(true_labels, prediction)
        
        conf = confusion_matrix(true_labels, prediction)
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)
        print(f"epoch: {epoch} train_loss: {mean_loss_train:.4f}, acc: {acc:.4f}")
        
        # Store average distillation losses for this epoch
        if len(epoch_dl1) > 0:
            distillation_losses['dl1'].append(np.mean(epoch_dl1))
        if len(epoch_dl2) > 0:
            distillation_losses['dl2'].append(np.mean(epoch_dl2))
        if len(epoch_dl3) > 0:
            distillation_losses['dl3'].append(np.mean(epoch_dl3))

        # Store KLD, BCE and beta for this epoch
        component_histories['kld'].append(np.mean(epoch_kld))
        component_histories['bce'].append(np.mean(epoch_bce))
        component_histories['beta'].append(beta)

        # Early stopping based on accuracy
        if early_stop_acc is not None and acc >= early_stop_acc:
            print(f"  -> Early stopping triggered! Accuracy {acc:.4f} >= target {early_stop_acc:.4f}")
            break

    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(stu.state_dict(), f'{save_dir}/brc_stu.pt')

    return total_loss, total_acc, conf, distillation_losses, component_histories


###########################################################################
# TESTING FUNCTION
###########################################################################

def find_optimal_threshold(y_true, y_proba, method='youden'):
    """
    Find the optimal classification threshold.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for class 1
        method: 'youden' (Youden's J statistic) or 'f1' (maximize F1 score)
        
    Returns:
        Optimal threshold value
    """
    if method == 'youden':
        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    elif method == 'f1':
        # Find threshold that maximizes F1 score
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return best_threshold
    
    elif method == 'balanced':
        # Find threshold that maximizes balanced accuracy
        best_bal_acc = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (y_proba >= threshold).astype(int)
            bal_acc = balanced_accuracy_score(y_true, preds)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = threshold
        return best_threshold
    
    else:
        return 0.5


def testing_stu(x1_test, x2_test, x3_test, num_class, layers_size, latent_dim,
                y_test, y_test_cat, batch_size, model_path='./checkpoints/brc_stu.pt',
                optimize_threshold=True, fusion_mode='vcdn'):
    """
    Test the trained student model.
    
    Key PyTorch translations:
    - model.eval() sets evaluation mode
    - torch.no_grad() context for inference (saves memory, faster)
    
    Args:
        x1_test, x2_test, x3_test: Test data for each view
        y_test: Ground truth labels
        y_test_cat: One-hot encoded labels
        model_path: Path to saved student model
        optimize_threshold: Whether to find and use optimal threshold
        
    Returns:
        Tuple of metrics: (f1, auc, predictions, balanced_acc, precision, recall, accuracy, optimal_threshold)
    """
    # Convert to tensors
    if isinstance(x1_test, np.ndarray):
        x1_tensor = torch.from_numpy(x1_test).float().to(device)
    else:
        x1_tensor = x1_test.float().to(device)
    
    if isinstance(x2_test, np.ndarray):
        x2_tensor = torch.from_numpy(x2_test).float().to(device)
    else:
        x2_tensor = x2_test.float().to(device)
    
    if isinstance(x3_test, np.ndarray):
        x3_tensor = torch.from_numpy(x3_test).float().to(device)
    else:
        x3_tensor = x3_test.float().to(device)
    
    # Create and load model
    stu = Student(x1_test, x2_test, x3_test, num_class, layers_size, latent_dim,
                  batch_size, fusion_mode=fusion_mode)
    stu = stu.to(device)
    stu.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode and disable gradients
    stu.eval()
    with torch.no_grad():
        (_, _, _, _, _, _, _, _, _, pred_labels_score) = stu(x1_tensor, x2_tensor, x3_tensor)
    
    # Convert to numpy for sklearn metrics
    pred_labels_np = pred_labels_score.cpu().numpy()
    y_proba = pred_labels_np[:, 1]  # Probability of class 1
    
    # Default predictions (threshold = 0.5)
    test_pred_default = np.argmax(pred_labels_np, axis=1)
    
    # Calculate AUC (threshold-independent)
    auc_kd_svae_vcdn = roc_auc_score(y_test, y_proba)
    
    # Calculate metrics with default threshold (0.5)
    acc_default = accuracy_score(y_test, test_pred_default)
    balanced_acc_default = balanced_accuracy_score(y_test, test_pred_default)
    f1_default = f1_score(y_test, test_pred_default)
    precision_default = precision_score(y_test, test_pred_default, zero_division=0)
    recall_default = recall_score(y_test, test_pred_default, zero_division=0)
    
    print("Test Results (Default Threshold = 0.5):")
    print(f"  Accuracy: {acc_default:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc_default:.4f}")
    print(f"  F1 Score: {f1_default:.4f}")
    print(f"  Precision: {precision_default:.4f}")
    print(f"  Recall: {recall_default:.4f}")
    print(f"  AUC: {auc_kd_svae_vcdn:.4f}")
    
    # Optimal threshold analysis
    optimal_threshold = 0.5
    if optimize_threshold:
        # Find optimal thresholds using different methods
        threshold_youden = find_optimal_threshold(y_test, y_proba, method='youden')
        threshold_balanced = find_optimal_threshold(y_test, y_proba, method='balanced')
        
        # Use the balanced accuracy threshold as primary
        optimal_threshold = threshold_balanced
        
        # Predictions with optimal threshold
        test_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics with optimal threshold
        acc_optimal = accuracy_score(y_test, test_pred_optimal)
        balanced_acc_optimal = balanced_accuracy_score(y_test, test_pred_optimal)
        f1_optimal = f1_score(y_test, test_pred_optimal)
        precision_optimal = precision_score(y_test, test_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_test, test_pred_optimal, zero_division=0)
        
        print(f"\nTest Results (Optimal Threshold = {optimal_threshold:.3f}):")
        print(f"  Accuracy: {acc_optimal:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc_optimal:.4f}")
        print(f"  F1 Score: {f1_optimal:.4f}")
        print(f"  Precision: {precision_optimal:.4f}")
        print(f"  Recall: {recall_optimal:.4f}")
        print(f"  AUC: {auc_kd_svae_vcdn:.4f}")
        
        print(f"\nThreshold Analysis:")
        print(f"  Youden's J optimal: {threshold_youden:.3f}")
        print(f"  Balanced Acc optimal: {threshold_balanced:.3f}")
        
        # Show probability distribution
        print(f"\nProbability Distribution:")
        print(f"  Class 0 (short-term) - mean: {y_proba[y_test == 0].mean():.3f}, "
              f"std: {y_proba[y_test == 0].std():.3f}")
        print(f"  Class 1 (long-term)  - mean: {y_proba[y_test == 1].mean():.3f}, "
              f"std: {y_proba[y_test == 1].std():.3f}")
        
        # Return optimal threshold metrics
        return (f1_optimal, auc_kd_svae_vcdn, y_proba, balanced_acc_optimal, 
                precision_optimal, recall_optimal, acc_optimal, optimal_threshold)
    
    return (f1_default, auc_kd_svae_vcdn, y_proba, balanced_acc_default, 
            precision_default, recall_default, acc_default, optimal_threshold)


###########################################################################
# PLOTTING FUNCTIONS
###########################################################################

def plot_distillation_losses(distillation_losses, save_path=None, title="Distillation Losses"):
    """
    Plot distillation losses over epochs.
    
    Args:
        distillation_losses: Dictionary with keys like 'dl1', 'dl2', 'dl3' and values as lists of losses
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if not distillation_losses or len(distillation_losses) == 0:
        print("Warning: No distillation losses to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(distillation_losses[list(distillation_losses.keys())[0]]) + 1)
    
    for key, losses in distillation_losses.items():
        if len(losses) > 0:
            plt.plot(epochs, losses, label=f'{key}', marker='o', linewidth=2, markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Distillation Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    
    plt.close()


def plot_roc_curves_cv(fold_roc_data, save_path=None, title="ROC Curves - Cross-Validation"):
    """
    Plot ROC curves for each fold with mean curve and confidence interval.
    
    Args:
        fold_roc_data: List of tuples (y_true, y_proba, auc) for each fold
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Define common FPR values for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    # Plot individual fold curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_roc_data)))
    
    for i, (y_true, y_proba, auc_score) in enumerate(fold_roc_data):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        aucs.append(auc_score)
        
        # Interpolate TPR at common FPR values
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Plot individual fold curve
        plt.plot(fpr, tpr, alpha=0.4, color=colors[i], 
                 label=f'Fold {i+1} (AUC = {auc_score:.2f})')
    
    # Calculate mean and std of TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='blue', linewidth=2.5,
             label=f'Mean ROC (AUC = {mean_auc:.2f} +/- {std_auc:.2f})')
    
    # Plot confidence interval (+-1 std)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2,
                     label='95% CI')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ROC plot to: {save_path}")
    
    plt.close()


def plot_loss_with_ci(fold_losses, save_path=None, title="Student Training Loss"):
    """
    Plot training loss across folds with confidence interval.
    
    Args:
        fold_losses: List of loss histories (one per fold)
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Pad losses to same length (in case of early stopping)
    max_epochs = max(len(losses) for losses in fold_losses)
    padded_losses = []
    
    for losses in fold_losses:
        if len(losses) < max_epochs:
            # Pad with last value (converged)
            padded = np.array(losses + [losses[-1]] * (max_epochs - len(losses)))
        else:
            padded = np.array(losses)
        padded_losses.append(padded)
    
    losses_array = np.array(padded_losses)
    epochs = np.arange(1, max_epochs + 1)
    
    # Calculate mean and std
    mean_loss = np.mean(losses_array, axis=0)
    std_loss = np.std(losses_array, axis=0)
    
    # Plot individual fold curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_losses)))
    for i, losses in enumerate(fold_losses):
        fold_epochs = np.arange(1, len(losses) + 1)
        plt.plot(fold_epochs, losses, alpha=0.4, color=colors[i], 
                 label=f'Fold {i+1}')
    
    # Plot mean curve
    plt.plot(epochs, mean_loss, color='blue', linewidth=2.5,
             label=f'Mean Loss')
    
    # Plot confidence interval
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                     color='blue', alpha=0.2, label='+-1 Std')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved loss plot to: {save_path}")
    
    plt.close()


def plot_accuracy_with_ci(fold_accuracies, save_path=None, title="Student Training Accuracy"):
    """
    Plot training accuracy across folds with confidence interval.
    
    Args:
        fold_accuracies: List of accuracy histories (one per fold)
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Pad accuracies to same length (in case of early stopping)
    max_epochs = max(len(accs) for accs in fold_accuracies)
    padded_accs = []
    
    for accs in fold_accuracies:
        if len(accs) < max_epochs:
            # Pad with last value (converged)
            padded = np.array(accs + [accs[-1]] * (max_epochs - len(accs)))
        else:
            padded = np.array(accs)
        padded_accs.append(padded)
    
    accs_array = np.array(padded_accs)
    epochs = np.arange(1, max_epochs + 1)
    
    # Calculate mean and std
    mean_acc = np.mean(accs_array, axis=0)
    std_acc = np.std(accs_array, axis=0)
    
    # Plot individual fold curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_accuracies)))
    for i, accs in enumerate(fold_accuracies):
        fold_epochs = np.arange(1, len(accs) + 1)
        plt.plot(fold_epochs, accs, alpha=0.4, color=colors[i], 
                 label=f'Fold {i+1}')
    
    # Plot mean curve
    plt.plot(epochs, mean_acc, color='green', linewidth=2.5,
             label=f'Mean Accuracy')
    
    # Plot confidence interval
    plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, 
                     color='green', alpha=0.2, label='+-1 Std')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved accuracy plot to: {save_path}")
    
    plt.close()


def plot_roc_curve_single(y_true, y_proba, auc_score, save_path=None, title="ROC Curve"):
    """
    Plot a single ROC curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for class 1
        auc_score: AUC score
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.plot(fpr, tpr, color='blue', linewidth=2.5,
             label=f'ROC curve (AUC = {auc_score:.2f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ROC plot to: {save_path}")
    
    plt.close()


def plot_training_curves(loss_history, acc_history, save_path=None, title="Student Training Curves"):
    """
    Plot training loss and accuracy on the same figure.
    
    Args:
        loss_history: List of loss values per epoch
        acc_history: List of accuracy values per epoch
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(loss_history) + 1)
    
    # Loss plot
    ax1.plot(epochs, loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, acc_history, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved training curves to: {save_path}")
    
    plt.close()


###########################################################################
# MAIN TRAINING PIPELINE
# Uncomment and adapt when preprocess module is available
###########################################################################

def run_training_pipeline():
    """
    Complete training pipeline for KD-SVAE-VCDN.
    
    This function demonstrates the 3-step training process:
    1. Train individual teachers on each omics modality
    2. Train combined teachers with knowledge distillation
    3. Train student model with knowledge from all teachers
    """
    
    # NOTE: These variables should come from your preprocess module:
    # x1, x2, x3 - individual omics data
    # y1, y2, y3 - labels for each omics
    # x_c_1_2, x_c_1_3, etc. - combined data for teacher level 2
    # x1_c, x2_c, x3_c - common samples across all omics
    # x1_test, x2_test, x3_test - test data
    
    # ################### STEP 1: Train Individual Teachers ###################
    
    # Teacher 1 (e.g., RNA-seq)
    # softs = training_te_level1(
    #     x1, num_class=2, layers_size_te1=layer_size_te1,
    #     layers_size_te2=layer_size_te2, lr=0.005,
    #     latent_dim_te1=latent_dim_te1, latent_dim_te2=latent_dim_te2,
    #     y=y1, y_cat=to_categorical(y1, 2),
    #     num_epoch=15, batch_size=te_batch_size, temperature=1.5,
    #     use_sample_weight=True, steps=1, num_te=1
    # )
    
    # Get soft labels for step 2
    # softs1_2_s2 = get_predictions(
    #     x_c_1_2, model_path='./checkpoints/KD_TE_11.pt',
    #     num_class=2, layers_size_te1=layer_size_te1, layers_size_te2=layer_size_te2,
    #     latent_dim_te1=latent_dim_te1, latent_dim_te2=latent_dim_te2,
    #     temperature=1.5, step=1
    # )
    
    # ... (repeat for other teachers)
    
    # ################### STEP 2: Train Combined Teachers ###################
    
    # Combine modalities and train with distillation
    # inputs1_2_s2 = torch.cat([x_c_1_2, x_c_2_1], dim=1)
    
    # softs2_1 = training_te_level2(
    #     inputs1_2_s2, num_class=2,
    #     layers_size_te1=layer_size_te1, layers_size_te2=layer_size_te2,
    #     lr=0.001, latent_dim_te1=latent_dim_te1, latent_dim_te2=latent_dim_te2,
    #     y=y_c_1_2, y_cat=to_categorical(y_c_1_2, 2),
    #     softs1=softs1_2_s2, softs2=softs2_1_s2, a=0.7, b=0.3,
    #     num_epoch=15, batch_size=te_batch_size, temperature=1.5,
    #     use_sample_weight=True, steps=2, num_te=1
    # )
    
    # ... (repeat for other combinations)
    
    # ################### STEP 3: Train Student ###################
    
    # hist_loss, hist_acc, last_conf = training_stu(
    #     data1=x1_c, data2=x2_c, data3=x3_c,
    #     num_class=2, layers_size=stu_layers_size, lr=0.001,
    #     stu_latent_dim=stu_latent_dim,
    #     y=y1_c, y_cat=to_categorical(y1_c, 2),
    #     softs1=softs1_2_s3, softs2=softs1_3_s3, softs3=softs2_3_s3,
    #     a=0.2, b=6.2, c=1.8,
    #     num_epoch=25, batch_size=stu_batch_size, temperature=1.5,
    #     use_sample_weight=True
    # )
    
    # ################### TESTING ###################
    
    # f1, auc, preds, bal_acc, prec, rec, acc = testing_stu(
    #     x1_test, x2_test, x3_test,
    #     num_class=2, layers_size=stu_layers_size, latent_dim=stu_latent_dim,
    #     y_test=y1_test, y_test_cat=to_categorical(y1_test, 2),
    #     batch_size=stu_batch_size
    # )
    
    pass


if __name__ == '__main__':
    print(f"Using device: {device}")
    # run_training_pipeline()

