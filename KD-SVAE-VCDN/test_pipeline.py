#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to verify the KD-SVAE-VCDN pipeline works.
Runs a few epochs on a small subset of data.

Usage:
    python test_pipeline.py
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import device, num_class
from preprocess import prepare_data_loaders, get_single_modality_data
from KD import Teacher, Student, loss_teacher_level1, loss_student, Clf
from VAEs import teacher1_Encoder, teacher1_Decoder, student_Encoder, student_Decoder
from train_test import training_te_level1, to_categorical


def test_data_loading():
    """Test data loading and preprocessing."""
    print("\n" + "=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return None
    
    try:
        data = prepare_data_loaders(
            data_dir=data_dir,
            test_size=0.2,
            batch_size=16,
            max_mirna_features=200,    # Small for testing
            max_rnaseq_features=300,   # Small for testing
            max_meth_features=200,     # Small for testing
        )
        print("\n✓ Data loading PASSED")
        return data
    except Exception as e:
        print(f"\n✗ Data loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_creation(data):
    """Test model instantiation."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Creation")
    print("=" * 60)
    
    try:
        # Get dimensions
        n_mirna = data['n_mirna_features']
        n_rnaseq = data['n_rnaseq_features']
        n_meth = data['n_meth_features']
        
        print(f"Feature dimensions: miRNA={n_mirna}, RNAseq={n_rnaseq}, Meth={n_meth}")
        
        # Test Teacher model
        print("\nCreating Teacher model...")
        layers_size = [256, 128, 64]
        latent_dim = 20
        
        teacher = Teacher(
            data=n_mirna,  # Can pass int instead of data
            num_class=num_class,
            layers_size_te1=layers_size,
            layers_size_te2=layers_size,
            latent_dim_te1=latent_dim,
            latent_dim_te2=latent_dim,
            temperature=1.5,
            step=1
        ).to(device)
        
        # Test forward pass
        x_test = torch.randn(4, n_mirna).to(device)
        recon, means, log_var, pred, soft = teacher(x_test)
        
        print(f"  Input shape: {x_test.shape}")
        print(f"  Recon shape: {recon.shape}")
        print(f"  Means shape: {means.shape}")
        print(f"  Predictions shape: {pred.shape}")
        
        # Test Student model
        print("\nCreating Student model...")
        stu_layers = [256, 128, 64]
        stu_latent = 10
        
        student = Student(
            data1=n_mirna,
            data2=n_rnaseq,
            data3=n_meth,
            num_class=num_class,
            stu_layers_size=stu_layers,
            stu_latent_dim=stu_latent,
            batch_size=16
        ).to(device)
        
        # Test forward pass
        x1 = torch.randn(4, n_mirna).to(device)
        x2 = torch.randn(4, n_rnaseq).to(device)
        x3 = torch.randn(4, n_meth).to(device)
        
        outputs = student(x1, x2, x3)
        print(f"  Student output count: {len(outputs)}")
        print(f"  Student predictions shape: {outputs[-1].shape}")
        
        print("\n✓ Model creation PASSED")
        return teacher, student
        
    except Exception as e:
        print(f"\n✗ Model creation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_training_loop(data):
    """Test a few training iterations."""
    print("\n" + "=" * 60)
    print("TEST 3: Training Loop (2 epochs)")
    print("=" * 60)
    
    try:
        # Get data
        x1_train = data['x1_train'].to(device)
        y_train = data['y_train']
        
        # Use small subset
        n_samples = min(100, len(x1_train))
        x1_train = x1_train[:n_samples]
        y_train = y_train[:n_samples]
        
        print(f"Training on {n_samples} samples...")
        
        # Create model
        n_features = x1_train.shape[1]
        layers_size = [128, 64, 32]
        latent_dim = 10
        
        teacher = Teacher(
            data=n_features,
            num_class=num_class,
            layers_size_te1=layers_size,
            layers_size_te2=layers_size,
            latent_dim_te1=latent_dim,
            latent_dim_te2=latent_dim,
            temperature=1.5,
            step=1
        ).to(device)
        
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
        
        # Convert labels to categorical
        y_cat = to_categorical(y_train, num_class).to(device)
        
        # Training loop
        batch_size = 16
        n_batches = (n_samples + batch_size - 1) // batch_size
        alpha = 0.01 * n_samples
        
        for epoch in range(2):
            epoch_loss = 0
            teacher.train()
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                x_batch = x1_train[start:end]
                y_batch = y_cat[start:end]
                
                # Sample weights (uniform for testing)
                sample_weight = torch.ones(end - start).to(device)
                
                optimizer.zero_grad()
                recon, means, log_var, pred, soft = teacher(x_batch)
                loss = loss_teacher_level1(
                    x_batch, recon, means, log_var,
                    y_batch, pred, sample_weight, alpha
                )
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"  Epoch {epoch + 1}: Loss = {epoch_loss / n_batches:.4f}")
        
        print("\n✓ Training loop PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Training loop FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(data):
    """Test PyTorch DataLoader iteration."""
    print("\n" + "=" * 60)
    print("TEST 4: DataLoader Iteration")
    print("=" * 60)
    
    try:
        train_loader = data['train_loader']
        
        print(f"Number of batches: {len(train_loader)}")
        
        # Get one batch
        for batch in train_loader:
            mirna, rnaseq, meth, labels = batch
            print(f"  Batch shapes:")
            print(f"    miRNA: {mirna.shape}")
            print(f"    RNAseq: {rnaseq.shape}")
            print(f"    Methylation: {meth.shape}")
            print(f"    Labels: {labels.shape}")
            break
        
        print("\n✓ DataLoader PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ DataLoader FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# KD-SVAE-VCDN PIPELINE TEST")
    print("#" * 60)
    print(f"\nUsing device: {device}")
    
    # Test 1: Data loading
    data = test_data_loading()
    if data is None:
        print("\n" + "!" * 60)
        print("Cannot proceed without data. Please check your data directory.")
        print("!" * 60)
        return
    
    # Test 2: Model creation
    teacher, student = test_model_creation(data)
    
    # Test 3: Training loop
    test_training_loop(data)
    
    # Test 4: DataLoader
    test_dataloader(data)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("All basic tests completed!")
    print("\nTo run full training, use:")
    print("  python train_test.py")
    print("\nOr import and use the training functions:")
    print("  from preprocess import prepare_data_loaders")
    print("  from train_test import training_te_level1, training_stu")


if __name__ == '__main__':
    main()


