# -*- coding: utf-8 -*-
"""
Data Loading module for KD-SVAE-VCDN.
Handles loading preprocessed data and creating PyTorch datasets.
Aligns modalities when creating datasets.

Author: Alberto Bastero
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json


###########################################################################
# MULTI-OMICS DATASET
###########################################################################

class MultiOmicsDataset(Dataset):
    """PyTorch Dataset for multi-omics data."""
    
    def __init__(self, mirna_data, rnaseq_data, meth_data, labels):
        """
        Args:
            mirna_data: miRNA features (numpy array)
            rnaseq_data: RNAseq features (numpy array)
            meth_data: DNA methylation features (numpy array)
            labels: Binary labels (numpy array)
        """
        self.mirna = torch.FloatTensor(mirna_data)
        self.rnaseq = torch.FloatTensor(rnaseq_data)
        self.meth = torch.FloatTensor(meth_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.mirna[idx],
            self.rnaseq[idx],
            self.meth[idx],
            self.labels[idx]
        )


###########################################################################
# LOAD PREPROCESSED DATA
###########################################################################

def load_preprocessed_data(preprocessed_dir):
    """
    Load preprocessed data from CSV files.
    Each modality is loaded separately with its own patient set.
    
    Args:
        preprocessed_dir: Directory containing train/ and test/ subdirectories
        
    Returns:
        Dictionary with train/test data and metadata
    """
    train_dir = os.path.join(preprocessed_dir, 'train')
    test_dir = os.path.join(preprocessed_dir, 'test')
    
    # Load metadata
    metadata_path = os.path.join(preprocessed_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load train data (each modality separately)
    print("Loading train data from preprocessed files...")
    mirna_train = None
    rnaseq_train = None
    meth_train = None
    labels_mirna_train = None
    labels_rnaseq_train = None
    labels_meth_train = None
    
    mirna_train_path = os.path.join(train_dir, 'mirna.csv')
    rnaseq_train_path = os.path.join(train_dir, 'rnaseq.csv')
    meth_train_path = os.path.join(train_dir, 'methylation.csv')
    
    if os.path.exists(mirna_train_path):
        mirna_train = pd.read_csv(mirna_train_path, index_col=0)
        labels_mirna_df = pd.read_csv(os.path.join(train_dir, 'labels_mirna.csv'))
        # Handle both 'patient_id' column and index column
        if 'patient_id' in labels_mirna_df.columns:
            labels_mirna_train = labels_mirna_df
        else:
            # First column is patient_id (index)
            labels_mirna_train = labels_mirna_df.rename(columns={labels_mirna_df.columns[0]: 'patient_id'})
        print(f"  Loaded miRNA train: {mirna_train.shape} ({len(labels_mirna_train)} labels)")
    
    if os.path.exists(rnaseq_train_path):
        rnaseq_train = pd.read_csv(rnaseq_train_path, index_col=0)
        labels_rnaseq_df = pd.read_csv(os.path.join(train_dir, 'labels_rnaseq.csv'))
        # Handle both 'patient_id' column and index column
        if 'patient_id' in labels_rnaseq_df.columns:
            labels_rnaseq_train = labels_rnaseq_df
        else:
            # First column is patient_id (index)
            labels_rnaseq_train = labels_rnaseq_df.rename(columns={labels_rnaseq_df.columns[0]: 'patient_id'})
        print(f"  Loaded RNAseq train: {rnaseq_train.shape} ({len(labels_rnaseq_train)} labels)")
    
    if os.path.exists(meth_train_path):
        meth_train = pd.read_csv(meth_train_path, index_col=0)
        labels_meth_path = os.path.join(train_dir, 'labels_methylation.csv')
        if os.path.exists(labels_meth_path):
            labels_meth_df = pd.read_csv(labels_meth_path)
            # Handle both 'patient_id' column and index column
            if 'patient_id' in labels_meth_df.columns:
                labels_meth_train = labels_meth_df
            else:
                # First column is patient_id (index)
                labels_meth_train = labels_meth_df.rename(columns={labels_meth_df.columns[0]: 'patient_id'})
            print(f"  Loaded methylation train: {meth_train.shape} ({len(labels_meth_train)} labels)")
        else:
            labels_meth_train = None
            print(f"  Loaded methylation train: {meth_train.shape} (no labels file found)")
    
    # Load test data (each modality separately)
    print("\nLoading test data from preprocessed files...")
    mirna_test = None
    rnaseq_test = None
    meth_test = None
    labels_mirna_test = None
    labels_rnaseq_test = None
    labels_meth_test = None
    
    mirna_test_path = os.path.join(test_dir, 'mirna.csv')
    rnaseq_test_path = os.path.join(test_dir, 'rnaseq.csv')
    meth_test_path = os.path.join(test_dir, 'methylation.csv')
    
    if os.path.exists(mirna_test_path):
        mirna_test = pd.read_csv(mirna_test_path, index_col=0)
        labels_mirna_df = pd.read_csv(os.path.join(test_dir, 'labels_mirna.csv'))
        # Handle both 'patient_id' column and index column
        if 'patient_id' in labels_mirna_df.columns:
            labels_mirna_test = labels_mirna_df
        else:
            # First column is patient_id (index)
            labels_mirna_test = labels_mirna_df.rename(columns={labels_mirna_df.columns[0]: 'patient_id'})
        print(f"  Loaded miRNA test: {mirna_test.shape} ({len(labels_mirna_test)} labels)")
    
    if os.path.exists(rnaseq_test_path):
        rnaseq_test = pd.read_csv(rnaseq_test_path, index_col=0)
        labels_rnaseq_df = pd.read_csv(os.path.join(test_dir, 'labels_rnaseq.csv'))
        # Handle both 'patient_id' column and index column
        if 'patient_id' in labels_rnaseq_df.columns:
            labels_rnaseq_test = labels_rnaseq_df
        else:
            # First column is patient_id (index)
            labels_rnaseq_test = labels_rnaseq_df.rename(columns={labels_rnaseq_df.columns[0]: 'patient_id'})
        print(f"  Loaded RNAseq test: {rnaseq_test.shape} ({len(labels_rnaseq_test)} labels)")
    
    if os.path.exists(meth_test_path):
        meth_test = pd.read_csv(meth_test_path, index_col=0)
        labels_meth_path = os.path.join(test_dir, 'labels_methylation.csv')
        if os.path.exists(labels_meth_path):
            labels_meth_df = pd.read_csv(labels_meth_path)
            # Handle both 'patient_id' column and index column
            if 'patient_id' in labels_meth_df.columns:
                labels_meth_test = labels_meth_df
            else:
                # First column is patient_id (index)
                labels_meth_test = labels_meth_df.rename(columns={labels_meth_df.columns[0]: 'patient_id'})
            print(f"  Loaded methylation test: {meth_test.shape} ({len(labels_meth_test)} labels)")
        else:
            labels_meth_test = None
            print(f"  Loaded methylation test: {meth_test.shape} (no labels file found)")
    
    return {
        'mirna_train': mirna_train,
        'rnaseq_train': rnaseq_train,
        'meth_train': meth_train,
        'labels_mirna_train': labels_mirna_train,
        'labels_rnaseq_train': labels_rnaseq_train,
        'labels_meth_train': labels_meth_train,
        'mirna_test': mirna_test,
        'rnaseq_test': rnaseq_test,
        'meth_test': meth_test,
        'labels_mirna_test': labels_mirna_test,
        'labels_rnaseq_test': labels_rnaseq_test,
        'labels_meth_test': labels_meth_test,
        'metadata': metadata,
    }


def align_modalities(mirna_df, rnaseq_df, meth_df, labels_dict, split='train'):
    """
    Align modalities by finding common patients and filling missing modalities with zeros.
    
    Args:
        mirna_df: miRNA DataFrame (patients x features)
        rnaseq_df: RNAseq DataFrame (patients x features)
        meth_df: Methylation DataFrame (patients x features)
        labels_dict: Dictionary with labels for each modality
        split: 'train' or 'test'
        
    Returns:
        Tuple of (aligned_mirna, aligned_rnaseq, aligned_meth, aligned_labels, patient_ids)
    """
    # Get all patient IDs from all modalities and labels
    all_patients = set()
    
    if mirna_df is not None:
        all_patients |= set(mirna_df.index)
    if rnaseq_df is not None:
        all_patients |= set(rnaseq_df.index)
    if meth_df is not None:
        all_patients |= set(meth_df.index)
    
    # Also include patients from labels
    # Handle both 'patient_id' and 'index' column names
    for mod in ['mirna', 'rnaseq', 'meth']:
        labels_key = f'labels_{mod}_{split}'
        if labels_dict.get(labels_key) is not None:
            labels_df = labels_dict[labels_key]
            if 'patient_id' in labels_df.columns:
                all_patients |= set(labels_df['patient_id'].values)
            elif 'index' in labels_df.columns:
                all_patients |= set(labels_df['index'].values)
            elif len(labels_df.columns) > 0:
                # First column is patient_id
                all_patients |= set(labels_df.iloc[:, 0].values)
    
    all_patients = sorted(list(all_patients))
    
    # Get feature dimensions
    n_mirna_features = mirna_df.shape[1] if mirna_df is not None else 0
    n_rnaseq_features = rnaseq_df.shape[1] if rnaseq_df is not None else 0
    n_meth_features = meth_df.shape[1] if meth_df is not None else 0
    
    # Align each modality (reindex to all patients, fill missing with 0)
    if mirna_df is not None:
        mirna_aligned = mirna_df.reindex(all_patients, fill_value=0)
    else:
        mirna_aligned = pd.DataFrame(0, index=all_patients, columns=range(n_mirna_features))
    
    if rnaseq_df is not None:
        rnaseq_aligned = rnaseq_df.reindex(all_patients, fill_value=0)
    else:
        rnaseq_aligned = pd.DataFrame(0, index=all_patients, columns=range(n_rnaseq_features))
    
    if meth_df is not None:
        meth_aligned = meth_df.reindex(all_patients, fill_value=0)
    else:
        meth_aligned = pd.DataFrame(0, index=all_patients, columns=range(n_meth_features))
    
    # Align labels - use union of all labels, prioritize non-NaN values
    labels_aligned = pd.DataFrame(index=all_patients, columns=['patient_id', 'label'])
    labels_aligned['patient_id'] = all_patients
    
    # Fill labels from each modality
    for mod in ['mirna', 'rnaseq', 'meth']:
        labels_key = f'labels_{mod}_{split}'
        if labels_dict.get(labels_key) is not None:
            labels_mod = labels_dict[labels_key].copy()
            # Handle different column names for patient_id
            if 'patient_id' in labels_mod.columns:
                labels_mod = labels_mod.set_index('patient_id')
            elif 'index' in labels_mod.columns:
                labels_mod = labels_mod.set_index('index')
            else:
                # First column is patient_id
                labels_mod = labels_mod.set_index(labels_mod.columns[0])
            
            for patient_id in all_patients:
                if patient_id in labels_mod.index:
                    if pd.isna(labels_aligned.loc[patient_id, 'label']):
                        labels_aligned.loc[patient_id, 'label'] = labels_mod.loc[patient_id, 'label']
    
    # Handle 'methylation' key (saved as 'methylation' not 'meth')
    if labels_dict.get(f'labels_methylation_{split}') is not None:
        labels_mod = labels_dict[f'labels_methylation_{split}'].copy()
        # Handle different column names for patient_id
        if 'patient_id' in labels_mod.columns:
            labels_mod = labels_mod.set_index('patient_id')
        elif 'index' in labels_mod.columns:
            labels_mod = labels_mod.set_index('index')
        else:
            # First column is patient_id
            labels_mod = labels_mod.set_index(labels_mod.columns[0])
        
        for patient_id in all_patients:
            if patient_id in labels_mod.index:
                if pd.isna(labels_aligned.loc[patient_id, 'label']):
                    labels_aligned.loc[patient_id, 'label'] = labels_mod.loc[patient_id, 'label']
    
    # Filter out patients without labels
    has_labels = ~labels_aligned['label'].isna()
    patients_with_labels = labels_aligned[has_labels].index.tolist()
    
    if len(patients_with_labels) < len(all_patients):
        print(f"  Filtered out {len(all_patients) - len(patients_with_labels)} patients without labels")
    
    # Filter to patients with labels
    mirna_aligned = mirna_aligned.loc[patients_with_labels]
    rnaseq_aligned = rnaseq_aligned.loc[patients_with_labels]
    meth_aligned = meth_aligned.loc[patients_with_labels]
    labels_aligned = labels_aligned.loc[patients_with_labels]
    
    y = labels_aligned['label'].values.astype(int)
    
    return mirna_aligned, rnaseq_aligned, meth_aligned, y, patients_with_labels


###########################################################################
# DATA LOADER PREPARATION
###########################################################################

def prepare_data_loaders(preprocessed_dir, batch_size=32):
    """
    Prepare complete data loaders for training and testing from preprocessed data.
    Aligns modalities when creating datasets.
    
    Args:
        preprocessed_dir: Directory containing preprocessed train/test data
        batch_size: Batch size for data loaders
        
    Returns:
        Dictionary with train/test data loaders and metadata
    """
    print("=" * 60)
    print("LOADING AND ALIGNING PREPROCESSED MULTI-OMICS DATA")
    print("=" * 60)
    
    if not os.path.exists(preprocessed_dir):
        raise ValueError(f"Preprocessed data directory not found: {preprocessed_dir}\n"
                        f"Please run preprocess.py first to preprocess the data.")
    
    # Load preprocessed data (each modality separately)
    data = load_preprocessed_data(preprocessed_dir)
    
    # Prepare labels dictionary for alignment
    labels_dict = {
        'labels_mirna_train': data.get('labels_mirna_train'),
        'labels_rnaseq_train': data.get('labels_rnaseq_train'),
        'labels_meth_train': data.get('labels_meth_train'),
        'labels_methylation_train': data.get('labels_meth_train'),  # Also check 'methylation' key
        'labels_mirna_test': data.get('labels_mirna_test'),
        'labels_rnaseq_test': data.get('labels_rnaseq_test'),
        'labels_meth_test': data.get('labels_meth_test'),
        'labels_methylation_test': data.get('labels_meth_test'),  # Also check 'methylation' key
    }
    
    # Align modalities for train set
    print("\n" + "-" * 40)
    print("Aligning modalities for train set...")
    mirna_train, rnaseq_train, meth_train, y_train, patient_ids_train = align_modalities(
        data['mirna_train'], data['rnaseq_train'], data['meth_train'],
        labels_dict, split='train'
    )
    
    # Align modalities for test set
    print("\n" + "-" * 40)
    print("Aligning modalities for test set...")
    mirna_test, rnaseq_test, meth_test, y_test, patient_ids_test = align_modalities(
        data['mirna_test'], data['rnaseq_test'], data['meth_test'],
        labels_dict, split='test'
    )
    
    # Convert to numpy arrays
    X_mirna_train = mirna_train.values
    X_rnaseq_train = rnaseq_train.values
    X_meth_train = meth_train.values
    
    X_mirna_test = mirna_test.values
    X_rnaseq_test = rnaseq_test.values
    X_meth_test = meth_test.values
    
    print(f"\nAligned data shapes:")
    print(f"  Train - miRNA: {X_mirna_train.shape}, RNAseq: {X_rnaseq_train.shape}, Methylation: {X_meth_train.shape}")
    print(f"  Test  - miRNA: {X_mirna_test.shape}, RNAseq: {X_rnaseq_test.shape}, Methylation: {X_meth_test.shape}")
    print(f"  Labels - Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Create datasets
    train_dataset = MultiOmicsDataset(
        X_mirna_train, X_rnaseq_train, X_meth_train, y_train
    )
    test_dataset = MultiOmicsDataset(
        X_mirna_test, X_rnaseq_test, X_meth_test, y_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nCreated data loaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    # Create individual modality tensors for teacher training
    result = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        # Individual modality data (for teacher training)
        'x1_train': torch.FloatTensor(X_mirna_train),  # miRNA
        'x2_train': torch.FloatTensor(X_rnaseq_train), # RNAseq
        'x3_train': torch.FloatTensor(X_meth_train),   # Methylation
        'y_train': y_train,
        'x1_test': torch.FloatTensor(X_mirna_test),
        'x2_test': torch.FloatTensor(X_rnaseq_test),
        'x3_test': torch.FloatTensor(X_meth_test),
        'y_test': y_test,
        # Metadata
        'n_mirna_features': X_mirna_train.shape[1],
        'n_rnaseq_features': X_rnaseq_train.shape[1],
        'n_meth_features': X_meth_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'patient_ids_train': patient_ids_train,
        'patient_ids_test': patient_ids_test,
    }
    
    print("\n" + "=" * 60)
    print("DATA LOADER PREPARATION COMPLETE")
    print("=" * 60)
    
    return result


###########################################################################
# KD-AWARE DATA PREPARATION
# Correctly handles per-modality patient subsets for the Ranjbari
# teacher-student knowledge distillation framework.
###########################################################################

def _extract_patient_ids_and_labels(data_df, labels_df):
    """Extract aligned patient IDs and integer labels from a modality."""
    if labels_df is None or data_df is None:
        return None, None, None
    labels = labels_df.copy()
    if 'patient_id' in labels.columns:
        labels = labels.set_index('patient_id')
    elif 'index' in labels.columns:
        labels = labels.set_index('index')
    else:
        labels = labels.set_index(labels.columns[0])
    common = data_df.index.intersection(labels.index)
    return data_df.loc[common], labels.loc[common, 'label'].values.astype(int), list(common)


def prepare_kd_data(preprocessed_dir):
    """
    Load data and build per-modality, pairwise, and complete-case subsets
    required by the Ranjbari KD-SVAE-VCDN framework.

    Returns a dict with keys:
        - 'single_{1,2,3}_{train,test}': (tensor, labels_np, patient_ids)
        - 'pair_{12,13,23}_{train,test}': (tensor_a, tensor_b, labels_np, patient_ids)
        - 'complete_{train,test}': (tensor1, tensor2, tensor3, labels_np, patient_ids)
        - feature counts
    """
    raw = load_preprocessed_data(preprocessed_dir)

    result = {}
    for split in ('train', 'test'):
        mirna_df = raw[f'mirna_{split}']
        rnaseq_df = raw[f'rnaseq_{split}']
        meth_df = raw[f'meth_{split}']
        labels_mirna = raw[f'labels_mirna_{split}']
        labels_rnaseq = raw[f'labels_rnaseq_{split}']
        labels_meth = raw[f'labels_meth_{split}']

        mirna_data, mirna_y, mirna_ids = _extract_patient_ids_and_labels(mirna_df, labels_mirna)
        rnaseq_data, rnaseq_y, rnaseq_ids = _extract_patient_ids_and_labels(rnaseq_df, labels_rnaseq)
        meth_data, meth_y, meth_ids = _extract_patient_ids_and_labels(meth_df, labels_meth)

        # --- Level 1: single modality sets ---
        result[f'single_1_{split}'] = (
            torch.FloatTensor(mirna_data.values), mirna_y, mirna_ids
        )
        result[f'single_2_{split}'] = (
            torch.FloatTensor(rnaseq_data.values), rnaseq_y, rnaseq_ids
        )
        result[f'single_3_{split}'] = (
            torch.FloatTensor(meth_data.values), meth_y, meth_ids
        )

        # --- Level 2: pairwise intersections ---
        for tag, ids_a, ids_b, df_a, df_b, name_a, name_b in [
            ('12', mirna_ids, rnaseq_ids, mirna_data, rnaseq_data, 'mirna', 'rnaseq'),
            ('13', mirna_ids, meth_ids, mirna_data, meth_data, 'mirna', 'meth'),
            ('23', rnaseq_ids, meth_ids, rnaseq_data, meth_data, 'rnaseq', 'meth'),
        ]:
            common = sorted(set(ids_a) & set(ids_b))
            a_vals = df_a.loc[common].values
            b_vals = df_b.loc[common].values
            # Labels come from either modality (they agree on shared patients)
            if name_a == 'mirna':
                y_common = mirna_y[np.isin(mirna_ids, common)]
            elif name_a == 'rnaseq':
                y_common = rnaseq_y[np.isin(rnaseq_ids, common)]
            else:
                y_common = meth_y[np.isin(meth_ids, common)]
            # Re-extract labels by reindexing to ensure correct order
            labels_source = raw[f'labels_{name_a}_{split}'].copy()
            if 'patient_id' in labels_source.columns:
                labels_source = labels_source.set_index('patient_id')
            elif 'index' in labels_source.columns:
                labels_source = labels_source.set_index('index')
            else:
                labels_source = labels_source.set_index(labels_source.columns[0])
            y_common = labels_source.loc[common, 'label'].values.astype(int)

            result[f'pair_{tag}_{split}'] = (
                torch.FloatTensor(a_vals),
                torch.FloatTensor(b_vals),
                y_common,
                common,
            )

        # --- Complete case: intersection of all three ---
        complete_ids = sorted(set(mirna_ids) & set(rnaseq_ids) & set(meth_ids))
        labels_source = raw[f'labels_rnaseq_{split}'].copy()
        if 'patient_id' in labels_source.columns:
            labels_source = labels_source.set_index('patient_id')
        elif 'index' in labels_source.columns:
            labels_source = labels_source.set_index('index')
        else:
            labels_source = labels_source.set_index(labels_source.columns[0])
        y_complete = labels_source.loc[complete_ids, 'label'].values.astype(int)

        result[f'complete_{split}'] = (
            torch.FloatTensor(mirna_data.loc[complete_ids].values),
            torch.FloatTensor(rnaseq_data.loc[complete_ids].values),
            torch.FloatTensor(meth_data.loc[complete_ids].values),
            y_complete,
            complete_ids,
        )

    result['n_mirna_features'] = raw['mirna_train'].shape[1] if raw['mirna_train'] is not None else 0
    result['n_rnaseq_features'] = raw['rnaseq_train'].shape[1] if raw['rnaseq_train'] is not None else 0
    result['n_meth_features'] = raw['meth_train'].shape[1] if raw['meth_train'] is not None else 0

    # Print summary
    print("\nKD data subsets (train):")
    for i, name in [(1, 'miRNA'), (2, 'RNAseq'), (3, 'Methylation')]:
        t, y, ids = result[f'single_{i}_train']
        print(f"  Level 1 Teacher {i} ({name}): {len(ids)} patients x {t.shape[1]} features")
    for tag, name in [('12', 'miRNA+RNAseq'), ('13', 'miRNA+Meth'), ('23', 'RNAseq+Meth')]:
        a, b, y, ids = result[f'pair_{tag}_train']
        print(f"  Level 2 Teacher {tag} ({name}): {len(ids)} patients x ({a.shape[1]}+{b.shape[1]}) features")
    t1, t2, t3, y, ids = result['complete_train']
    print(f"  Student (complete cases): {len(ids)} patients x ({t1.shape[1]}+{t2.shape[1]}+{t3.shape[1]}) features")

    return result
