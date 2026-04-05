# -*- coding: utf-8 -*-
"""
Knowledge Distillation module for Multi-Omics integration
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

from VAEs import *
from config import *


###############################################################################
# FOCAL LOSS FOR CLASS IMBALANCE
###############################################################################

def focal_loss(pred_labels, y_cat, alpha=None, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Class imbalance is handled solely via the alpha (class weights) parameter,
    which should be set to inverse class frequency (computed by class_weights_for).
    
    Args:
        pred_labels: Predicted probabilities (N, C)
        y_cat: One-hot encoded targets (N, C)
        alpha: Class weights tensor (C,) or None for uniform
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        Focal loss value
    """
    p_t = (pred_labels * y_cat).sum(dim=1)
    focal_term = (1 - p_t) ** gamma
    ce = -torch.log(p_t + 1e-8)
    focal = focal_term * ce

    if alpha is not None:
        alpha_t = (alpha.unsqueeze(0) * y_cat).sum(dim=1)
        focal = alpha_t * focal
    
    return focal.mean()


################################## CLASSIFIER ##################################

class Clf(nn.Module):
    """Classifier head for VAE latent space."""
    
    def __init__(self, num_class, latent_dim, hidden_dim=64, temperature=1.0):
        super().__init__()
        self.num_class = num_class
        self.temperature = temperature
        
        # Input is concatenation of means and log_var (latent_dim * 2)
        input_dim = latent_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_class),
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        pred_labels = torch.softmax(logits, dim=-1)
        soft_labels = torch.softmax(logits / self.temperature, dim=-1)
        return pred_labels, soft_labels


################################## VCDN CLASSIFIER ##################################

class VCDN_Clf(nn.Module):
    """View Correlation Discovery Network classifier."""

    def __init__(self, num_view, num_class, latent_dim, layers_size):
        super().__init__()
        self.num_view = num_view
        self.num_class = num_class
        self.latent_dim = latent_dim

        # Per-view classifiers (output logits, no softmax)
        self.view_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 2, layers_size[-1]),  # mean + log_var
                nn.ReLU(),
                nn.Linear(layers_size[-1], num_class),
            ) for _ in range(num_view)
        ])

        # Fusion layer: takes the outer product of view logits
        # Output dimension is num_class^num_view
        fusion_input_dim = num_class ** num_view
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 2, num_class),
        )

    def forward(self, means_list, log_vars_list):
        """
        Args:
            means_list: List of mean tensors from each view encoder
            log_vars_list: List of log_var tensors from each view encoder
        """
        view_logits = []
        for i, (mean, log_var) in enumerate(zip(means_list, log_vars_list)):
            x = torch.cat([mean, log_var], dim=-1)
            logits = self.view_classifiers[i](x)
            view_logits.append(logits)

        # Compute outer product of view logits
        # For 3 views with 2 classes each: batch_size x 2 x 2 x 2 -> batch_size x 8
        fusion_input = view_logits[0]
        for logits in view_logits[1:]:
            fusion_input = torch.einsum('bi,bj->bij', fusion_input, logits)
            fusion_input = fusion_input.view(fusion_input.shape[0], -1)

        # Single softmax at the end to produce probabilities
        return torch.softmax(self.fusion(fusion_input), dim=-1)


class Concat_Clf(nn.Module):
    """Concatenation-based fusion classifier.

    Concatenates the latent representations (mean + log_var) from all views
    and classifies directly on the combined vector.  This avoids the outer-
    product bottleneck of the VCDN while keeping a comparable number of
    parameters.
    """

    def __init__(self, num_view, num_class, latent_dim, layers_size):
        super().__init__()
        # Input: [mean_1, log_var_1, mean_2, log_var_2, mean_3, log_var_3]
        input_dim = num_view * latent_dim * 2
        hidden_dim = layers_size[-1]

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_class),
        )

    def forward(self, means_list, log_vars_list):
        """
        Args:
            means_list: List of mean tensors from each view encoder
            log_vars_list: List of log_var tensors from each view encoder
        """
        parts = []
        for mean, log_var in zip(means_list, log_vars_list):
            parts.append(mean)
            parts.append(log_var)
        x = torch.cat(parts, dim=-1)
        return torch.softmax(self.classifier(x), dim=-1)


########################################## TEACHERS ##################################

class Teacher(nn.Module):
    """Teacher model with VAE encoder/decoder and classifier."""
    
    def __init__(self, data, num_class, layers_size_te1, layers_size_te2,
                 latent_dim_te1, latent_dim_te2, temperature, step=1):
        super().__init__()
        
        self.data = data
        self.step = step
        self.latent_dim_te1 = latent_dim_te1
        self.latent_dim_te2 = latent_dim_te2
        self.num_class = num_class
        self.layers_size_te1 = layers_size_te1
        self.layers_size_te2 = layers_size_te2
        self.temperature = temperature
        
        if step == 1:
            self.latent_dim = latent_dim_te1
            self.layers_size = layers_size_te1
            self.encoder = teacher1_Encoder(data, latent_dim_te1, layers_size_te1)
            self.decoder = teacher1_Decoder(data, latent_dim_te1, layers_size_te1)
        else:
            self.latent_dim = latent_dim_te2
            self.layers_size = layers_size_te2
            self.encoder = teacher2_Encoder(data, latent_dim_te2, layers_size_te2)
            self.decoder = teacher2_Decoder(data, latent_dim_te2, layers_size_te2)
            
        self.classifier = Clf(num_class, self.latent_dim, hidden_dim=64, temperature=temperature)
        
    def reparameterize(self, means, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + means
    
    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        
        # Classifier input is concatenation of means and log_var
        classifier_inputs = torch.cat([means, log_var], dim=1)
        pred_labels, soft_labels = self.classifier(classifier_inputs)
        
        recon = self.decoder(z)
        
        return recon, means, log_var, pred_labels, soft_labels


def loss_teacher_level1(data_tensor, recon, means, log_var, y_cat, pred_labels,
                        class_weights=None, gamma=2.0, beta=0.1):
    """
    Loss function for level 1 teacher (no knowledge distillation).

    Args:
        data_tensor: Original input data
        recon: Reconstructed data from decoder
        means: Mean of latent distribution
        log_var: Log variance of latent distribution
        y_cat: One-hot encoded labels
        pred_labels: Predicted class probabilities
        class_weights: Class weights for focal loss (C,) or None
        gamma: Focal loss gamma parameter (default: 2.0)
        beta: Weight for KL divergence term (default: 0.1)

    Returns:
        Combined loss value
    """
    clf_loss = focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)
    BCE = F.binary_cross_entropy(recon, data_tensor, reduction='mean')

    kl_loss = -0.5 * (1 + log_var - means.pow(2) - log_var.exp())
    KLD = kl_loss.sum(dim=1).mean()

    return BCE + beta * KLD + 5.0 * clf_loss


def loss_teacher_level2(data_tensor, recon, means, log_var, y_cat, pred_labels,
                        soft_labels1, soft_labels2, a, b,
                        class_weights=None, gamma=2.0, beta=0.1):
    """
    Loss function for level 2 teacher (with knowledge distillation from other teachers).

    Args:
        data_tensor: Original input data
        recon: Reconstructed data from decoder
        means: Mean of latent distribution
        log_var: Log variance of latent distribution
        y_cat: One-hot encoded labels
        pred_labels: Predicted class probabilities
        soft_labels1: Soft labels from teacher 1
        soft_labels2: Soft labels from teacher 2
        a: Weight for distillation loss from teacher 1
        b: Weight for distillation loss from teacher 2
        class_weights: Class weights for focal loss (C,) or None
        gamma: Focal loss gamma parameter (default: 2.0)
        beta: Weight for KL divergence term (default: 0.1)

    Returns:
        Combined loss value
    """
    clf_loss = focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)
    BCE = F.binary_cross_entropy(recon, data_tensor, reduction='mean')

    kl_loss = -0.5 * (1 + log_var - means.pow(2) - log_var.exp())
    KLD = kl_loss.sum(dim=1).mean()

    dl1 = F.kl_div(pred_labels.log(), soft_labels1, reduction='batchmean')
    dl2 = F.kl_div(pred_labels.log(), soft_labels2, reduction='batchmean')

    total_loss = BCE + beta * KLD + 10 * clf_loss + 2.0 * a * dl1 + 2.0 * b * dl2
    
    return total_loss, {
        'bce': BCE.item(),
        'kld': KLD.item(),
        'clf': clf_loss.item(),
        'dl1': dl1.item(),
        'dl2': dl2.item(),
        'total': total_loss.item()
    }


########################################## STUDENT #################################

class Student(nn.Module):
    """Student model with multiple view encoders/decoders and fusion classifier."""

    def __init__(self, data1, data2, data3, num_class, stu_layers_size,
                 stu_latent_dim, batch_size, num_view=3, fusion_mode='vcdn'):
        super().__init__()

        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.stu_latent_dim = stu_latent_dim
        self.num_class = num_class
        self.stu_layers_size = stu_layers_size
        self.batch_size = batch_size
        self.num_view = num_view
        self.fusion_mode = fusion_mode

        # View 1 encoder/decoder
        self.encoder1 = student_Encoder(data1, stu_latent_dim, stu_layers_size)
        self.decoder1 = student_Decoder(data1, stu_latent_dim, stu_layers_size)

        # View 2 encoder/decoder
        self.encoder2 = student_Encoder(data2, stu_latent_dim, stu_layers_size)
        self.decoder2 = student_Decoder(data2, stu_latent_dim, stu_layers_size)

        # View 3 encoder/decoder
        self.encoder3 = student_Encoder(data3, stu_latent_dim, stu_layers_size)
        self.decoder3 = student_Decoder(data3, stu_latent_dim, stu_layers_size)

        # Fusion classifier
        if fusion_mode == 'concat':
            self.vcdn = Concat_Clf(num_view, num_class, stu_latent_dim, stu_layers_size)
        else:
            self.vcdn = VCDN_Clf(num_view, num_class, stu_latent_dim, stu_layers_size)
        
    def reparameterize(self, means, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + means
    
    def forward(self, data1, data2, data3):
        # Encode each view
        means_1, log_var_1 = self.encoder1(data1)
        means_2, log_var_2 = self.encoder2(data2)
        means_3, log_var_3 = self.encoder3(data3)
        
        # Reparameterize
        z_1 = self.reparameterize(means_1, log_var_1)
        z_2 = self.reparameterize(means_2, log_var_2)
        z_3 = self.reparameterize(means_3, log_var_3)
        
        # VCDN classification
        vcdn_means = [means_1, means_2, means_3]
        vcdn_log_vars = [log_var_1, log_var_2, log_var_3]
        pred_labels = self.vcdn(vcdn_means, vcdn_log_vars)
        
        # Decode each view
        recon_1 = self.decoder1(z_1)
        recon_2 = self.decoder2(z_2)
        recon_3 = self.decoder3(z_3)
        
        return (recon_1, recon_2, recon_3,
                means_1, means_2, means_3,
                log_var_1, log_var_2, log_var_3,
                pred_labels)


def loss_student(data1, data2, data3, recon_1, recon_2, recon_3,
                 means_1, means_2, means_3, log_var_1, log_var_2, log_var_3,
                 pred_labels, y_cat_train, softs1, softs2, softs3,
                 a, b, c, class_weights=None, gamma=2.0, beta=0.1):
    """
    Loss function for student model with knowledge distillation from teachers.

    Args:
        data1, data2, data3: Original input data for each view
        recon_1, recon_2, recon_3: Reconstructed data for each view
        means_1, means_2, means_3: Means of latent distributions
        log_var_1, log_var_2, log_var_3: Log variances of latent distributions
        pred_labels: Predicted class probabilities from student
        y_cat_train: One-hot encoded labels
        softs1, softs2, softs3: Soft labels from teachers
        a, b, c: Weights for distillation losses from each teacher
        class_weights: Class weights for focal loss (C,) or None
        gamma: Focal loss gamma parameter (default: 2.0)
        beta: Weight for KL divergence term (default: 0.1)

    Returns:
        Combined loss value
    """
    clf_loss = focal_loss(pred_labels, y_cat_train, alpha=class_weights, gamma=gamma)
    
    rec_1_loss = F.binary_cross_entropy(recon_1, data1, reduction='mean')
    rec_2_loss = F.binary_cross_entropy(recon_2, data2, reduction='mean')
    rec_3_loss = F.binary_cross_entropy(recon_3, data3, reduction='mean')
    BCE = (rec_1_loss + rec_2_loss + rec_3_loss) / 3
    
    kl_1_loss = -0.5 * (1 + log_var_1 - means_1.pow(2) - log_var_1.exp())
    kl_1_loss = kl_1_loss.sum(dim=1).mean()
    
    kl_2_loss = -0.5 * (1 + log_var_2 - means_2.pow(2) - log_var_2.exp())
    kl_2_loss = kl_2_loss.sum(dim=1).mean()
    
    kl_3_loss = -0.5 * (1 + log_var_3 - means_3.pow(2) - log_var_3.exp())
    kl_3_loss = kl_3_loss.sum(dim=1).mean()
    
    KLD = (kl_1_loss + kl_2_loss + kl_3_loss) / 3
    
    dl1 = F.kl_div(pred_labels.log(), softs1, reduction='batchmean')
    dl2 = F.kl_div(pred_labels.log(), softs2, reduction='batchmean')
    dl3 = F.kl_div(pred_labels.log(), softs3, reduction='batchmean')
    
    total_loss = BCE + beta * KLD + 10 * clf_loss + 2.0 * (a * dl1 + b * dl2 + c * dl3)
    
    return total_loss, {
        'bce': BCE.item(),
        'kld': KLD.item(),
        'clf': clf_loss.item(),
        'dl1': dl1.item(),
        'dl2': dl2.item(),
        'dl3': dl3.item(),
        'total': total_loss.item()
    }

