# -*- coding: utf-8 -*-
"""
Top-Down Knowledge Distillation module for Multi-Omics integration.

Architecture: all modalities → pairwise → single-modality → integration
  Level 1: One teacher trained on all 3 modalities
  Level 2: Three pairwise teachers distilled from L1
  Level 3: Three single-modality students distilled from L2
  Integration: Fuses the 3 students' latent representations

Author: Alberto Bastero
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from VAEs import student_Encoder, student_Decoder
from KD import focal_loss, VCDN_Clf, Concat_Clf
from config import device


###############################################################################
# LEVEL 1 — ALL-MODALITY TEACHER
###############################################################################

class TopDownTeacherL1(nn.Module):
    """Level 1 teacher for top-down KD, trained on all 3 modalities.

    Uses 3 per-view VAE encoders/decoders with a fusion classifier that
    exposes pre-softmax logits so we can produce temperature-scaled soft
    labels for downstream L2 distillation.
    """

    def __init__(self, data1, data2, data3, num_class, layers_size,
                 latent_dim, temperature, num_view=3, fusion_mode='vcdn'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_class = num_class
        self.temperature = temperature
        self.num_view = num_view
        self.fusion_mode = fusion_mode

        self.encoder1 = student_Encoder(data1, latent_dim, layers_size)
        self.decoder1 = student_Decoder(data1, latent_dim, layers_size)
        self.encoder2 = student_Encoder(data2, latent_dim, layers_size)
        self.decoder2 = student_Decoder(data2, latent_dim, layers_size)
        self.encoder3 = student_Encoder(data3, latent_dim, layers_size)
        self.decoder3 = student_Decoder(data3, latent_dim, layers_size)

        if fusion_mode == 'concat':
            input_dim = num_view * latent_dim * 2
            hidden_dim = layers_size[-1]
            self.fusion_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_class),
            )
        else:
            self.view_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim * 2, layers_size[-1]),
                    nn.ReLU(),
                    nn.Linear(layers_size[-1], num_class),
                ) for _ in range(num_view)
            ])
            fusion_input_dim = num_class ** num_view
            self.fusion_net = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_input_dim // 2, num_class),
            )

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + means

    def _compute_logits(self, means_list, log_vars_list):
        """Pre-softmax fusion logits."""
        if self.fusion_mode == 'concat':
            parts = []
            for mean, log_var in zip(means_list, log_vars_list):
                parts.append(mean)
                parts.append(log_var)
            return self.fusion_net(torch.cat(parts, dim=-1))

        view_logits = []
        for i, (mean, log_var) in enumerate(zip(means_list, log_vars_list)):
            x = torch.cat([mean, log_var], dim=-1)
            view_logits.append(self.view_classifiers[i](x))
        fusion_input = view_logits[0]
        for logits in view_logits[1:]:
            fusion_input = torch.einsum('bi,bj->bij', fusion_input, logits)
            fusion_input = fusion_input.view(fusion_input.shape[0], -1)
        return self.fusion_net(fusion_input)

    def forward(self, data1, data2, data3):
        means_1, log_var_1 = self.encoder1(data1)
        means_2, log_var_2 = self.encoder2(data2)
        means_3, log_var_3 = self.encoder3(data3)

        z_1 = self.reparameterize(means_1, log_var_1)
        z_2 = self.reparameterize(means_2, log_var_2)
        z_3 = self.reparameterize(means_3, log_var_3)

        recon_1 = self.decoder1(z_1)
        recon_2 = self.decoder2(z_2)
        recon_3 = self.decoder3(z_3)

        logits = self._compute_logits(
            [means_1, means_2, means_3],
            [log_var_1, log_var_2, log_var_3],
        )
        pred_labels = torch.softmax(logits, dim=-1)
        soft_labels = torch.softmax(logits / self.temperature, dim=-1)

        return (recon_1, recon_2, recon_3,
                means_1, means_2, means_3,
                log_var_1, log_var_2, log_var_3,
                pred_labels, soft_labels)


###############################################################################
# INTEGRATION MODULE
###############################################################################

class IntegrationModule(nn.Module):
    """Fuses latent representations from 3 single-modality L3 students.

    Takes (means, log_vars) from each student and applies VCDN or concat
    fusion to produce a final classification.
    """

    def __init__(self, num_class, latent_dim, layers_size,
                 num_view=3, fusion_mode='vcdn'):
        super().__init__()
        self.num_class = num_class
        self.latent_dim = latent_dim
        self.fusion_mode = fusion_mode

        if fusion_mode == 'concat':
            self.fusion = Concat_Clf(num_view, num_class, latent_dim, layers_size)
        else:
            self.fusion = VCDN_Clf(num_view, num_class, latent_dim, layers_size)

    def forward(self, means_list, log_vars_list):
        return self.fusion(means_list, log_vars_list)


###############################################################################
# LOSS FUNCTIONS
###############################################################################

def loss_topdown_l1(data1, data2, data3,
                    recon_1, recon_2, recon_3,
                    means_1, means_2, means_3,
                    log_var_1, log_var_2, log_var_3,
                    pred_labels, y_cat,
                    class_weights=None, gamma=2.0, beta=0.1):
    """Loss for top-down L1 teacher (multi-view reconstruction, no KD)."""
    clf_loss = focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)

    bce_1 = F.binary_cross_entropy(recon_1, data1, reduction='mean')
    bce_2 = F.binary_cross_entropy(recon_2, data2, reduction='mean')
    bce_3 = F.binary_cross_entropy(recon_3, data3, reduction='mean')
    BCE = (bce_1 + bce_2 + bce_3) / 3

    kl_1 = (-0.5 * (1 + log_var_1 - means_1.pow(2) - log_var_1.exp())).sum(1).mean()
    kl_2 = (-0.5 * (1 + log_var_2 - means_2.pow(2) - log_var_2.exp())).sum(1).mean()
    kl_3 = (-0.5 * (1 + log_var_3 - means_3.pow(2) - log_var_3.exp())).sum(1).mean()
    KLD = (kl_1 + kl_2 + kl_3) / 3

    total = BCE + beta * KLD + 5.0 * clf_loss
    return total, {
        'bce': BCE.item(), 'kld': KLD.item(),
        'clf': clf_loss.item(), 'total': total.item(),
    }


def loss_topdown_l2(data_tensor, recon, means, log_var, y_cat, pred_labels,
                    soft_labels_l1, kd_weight,
                    class_weights=None, gamma=2.0, beta=0.1):
    """Loss for top-down L2 pairwise teacher (KD from 1 L1 teacher)."""
    clf_loss = focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)
    BCE = F.binary_cross_entropy(recon, data_tensor, reduction='mean')

    kl_loss = -0.5 * (1 + log_var - means.pow(2) - log_var.exp())
    KLD = kl_loss.sum(dim=1).mean()

    dl = F.kl_div(pred_labels.log(), soft_labels_l1, reduction='batchmean')

    total = BCE + beta * KLD + 10 * clf_loss + 2.0 * kd_weight * dl
    return total, {
        'bce': BCE.item(), 'kld': KLD.item(), 'clf': clf_loss.item(),
        'dl': dl.item(), 'total': total.item(),
    }


def loss_topdown_l3(data_tensor, recon, means, log_var, y_cat, pred_labels,
                    soft_labels_l2a, soft_labels_l2b, a, b,
                    class_weights=None, gamma=2.0, beta=0.1):
    """Loss for top-down L3 single-modality student (KD from 2 L2 teachers)."""
    clf_loss = focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)
    BCE = F.binary_cross_entropy(recon, data_tensor, reduction='mean')

    kl_loss = -0.5 * (1 + log_var - means.pow(2) - log_var.exp())
    KLD = kl_loss.sum(dim=1).mean()

    dl1 = F.kl_div(pred_labels.log(), soft_labels_l2a, reduction='batchmean')
    dl2 = F.kl_div(pred_labels.log(), soft_labels_l2b, reduction='batchmean')

    total = BCE + beta * KLD + 10 * clf_loss + 2.0 * (a * dl1 + b * dl2)
    return total, {
        'bce': BCE.item(), 'kld': KLD.item(), 'clf': clf_loss.item(),
        'dl1': dl1.item(), 'dl2': dl2.item(), 'total': total.item(),
    }


def loss_integration(pred_labels, y_cat, class_weights=None, gamma=2.0):
    """Loss for integration module (classification only)."""
    return focal_loss(pred_labels, y_cat, alpha=class_weights, gamma=gamma)
