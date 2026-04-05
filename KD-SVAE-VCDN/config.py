# -*- coding: utf-8 -*-
"""
Configuration file for KD-SVAE-VCDN model hyperparameters.

These layer sizes are designed for high-dimensional multi-omics data:
- miRNA: ~1000-2000 features
- RNAseq: ~40000-60000 features
- DNA Methylation: ~10000-35000 features
"""
import torch

### Hyperparameters ###

# Teacher 1 architecture (for single modalities)
# Progressive reduction: input -> 1024 -> 512 -> 256 -> latent
layer_size_te1 = [1024, 512, 256]
latent_size_te1 = [1024, 512, 256]  # For compatibility
latent_dim_te1 = 64  # Latent dimension for step 1 teachers

# Teacher 2 architecture (for combined modalities)
# Larger capacity for combined inputs
layer_size_te2 = [2048, 1024, 512]
latent_size_te2 = [2048, 1024, 512]  # For compatibility
latent_dim_te2 = 128  # Larger latent for combined modalities

# Student architecture
# Handles all 3 modalities simultaneously
stu_layers_size = [1024, 512, 256]
stu_latent_dim = 64  # Per-view latent dimension

# Model parameters
num_class = 2
num_view = 3
te_batch_size = 32
stu_batch_size = 32

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
default_lr = 0.001
default_te_epochs = 15
default_stu_epochs = 25
temperature = 1.5
