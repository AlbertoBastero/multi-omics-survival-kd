# -*- coding: utf-8 -*-
"""
VAE Encoders and Decoders for KD-SVAE-VCDN model.

Author: Alberto Bastero
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

from config import *


################################## ENCODERS ##################################

class teacher1_Encoder(nn.Module):
    """Encoder for Teacher 1 VAE (Level 1 single-modality teachers)."""
    
    def __init__(self, data, latent_dim_te1, layer_size_te1, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim_te1
        
        input_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer_size_te1[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te1[0], layer_size_te1[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te1[1], layer_size_te1[2]),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(layer_size_te1[2], latent_dim_te1)
        self.z_log_var = nn.Linear(layer_size_te1[2], latent_dim_te1)

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class teacher2_Encoder(nn.Module):
    """Encoder for Teacher 2 VAE (Level 2 pairwise teachers)."""
    
    def __init__(self, data, latent_dim_te2, layer_size_te2, dropout=0.3):
        super().__init__()
        self.latent_dim = latent_dim_te2
        
        input_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer_size_te2[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te2[0], layer_size_te2[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te2[1], layer_size_te2[2]),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(layer_size_te2[2], latent_dim_te2)
        self.z_log_var = nn.Linear(layer_size_te2[2], latent_dim_te2)

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class student_Encoder(nn.Module):
    """Encoder for Student VAE (per-view encoder in the student model)."""
    
    def __init__(self, data, stu_latent_dim, stu_layers_size, dropout=0.3):
        super().__init__()
        self.latent_dim = stu_latent_dim
        
        input_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, stu_layers_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(stu_layers_size[0], stu_layers_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(stu_layers_size[1], stu_layers_size[2]),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(stu_layers_size[2], stu_latent_dim)
        self.z_log_var = nn.Linear(stu_layers_size[2], stu_latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


################################## DECODERS ##################################

class teacher1_Decoder(nn.Module):
    """Decoder for Teacher 1 VAE."""
    
    def __init__(self, data, latent_dim_te1, layer_size_te1, dropout=0.3):
        super().__init__()
        
        output_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_te1, layer_size_te1[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te1[2], layer_size_te1[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te1[1], layer_size_te1[0]),
            nn.ReLU(),
            nn.Linear(layer_size_te1[0], output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class teacher2_Decoder(nn.Module):
    """Decoder for Teacher 2 VAE."""
    
    def __init__(self, data, latent_dim_te2, layer_size_te2, dropout=0.3):
        super().__init__()
        
        output_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_te2, layer_size_te2[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te2[2], layer_size_te2[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size_te2[1], layer_size_te2[0]),
            nn.ReLU(),
            nn.Linear(layer_size_te2[0], output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class student_Decoder(nn.Module):
    """Decoder for Student VAE."""
    
    def __init__(self, data, stu_latent_dim, stu_layers_size, dropout=0.3):
        super().__init__()
        
        output_dim = data.shape[1] if hasattr(data, 'shape') else data
        
        self.decoder = nn.Sequential(
            nn.Linear(stu_latent_dim, stu_layers_size[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(stu_layers_size[2], stu_layers_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(stu_layers_size[1], stu_layers_size[0]),
            nn.ReLU(),
            nn.Linear(stu_layers_size[0], output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
