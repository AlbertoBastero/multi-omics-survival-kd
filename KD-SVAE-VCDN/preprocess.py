# -*- coding: utf-8 -*-
"""
Data Preprocessing module for KD-SVAE-VCDN.
Handles loading and preprocessing of multi-omics BRCA data.

Author: Alberto Bastero
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


###########################################################################
# UTILITY FUNCTIONS
###########################################################################


def standardize_patient_id(patient_id):
    """
    Standardize patient IDs to a common format (with hyphens).
    Handles both TCGA.XX.XXXX and TCGA-XX-XXXX formats.

    Args:
        patient_id: Patient ID string

    Returns:
        Standardized patient ID with hyphens
    """
    return patient_id.replace(".", "-")


def normalize_data(data, method="minmax"):
    """
    Normalize data using specified method.

    Args:
        data: numpy array or pandas DataFrame
        method: 'minmax' or 'standard'

    Returns:
        Normalized data and fitted scaler
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    if isinstance(data, pd.DataFrame):
        normalized = pd.DataFrame(
            scaler.fit_transform(data), index=data.index, columns=data.columns
        )
    else:
        normalized = scaler.fit_transform(data)

    return normalized, scaler


def remove_low_variance_features(data, threshold=0.02):
    """
    Remove features with variance below threshold.

    Args:
        data: pandas DataFrame (samples x features)
        threshold: Variance threshold

    Returns:
        Filtered data and selector
    """
    # Ensure all column names are strings
    data.columns = data.columns.astype(str)

    selector = VarianceThreshold(threshold=threshold)

    # Fit on the data
    selector.fit(data)

    # Get selected feature indices
    selected_features = data.columns[selector.get_support()]

    # Return filtered data
    return data[selected_features], selector


def anova_feature_selection(data, labels, p_value_threshold=0.05):
    """
    Select features using ANOVA F-test.
    Features with p-value > threshold are discarded.

    Args:
        data: pandas DataFrame (samples x features)
        labels: numpy array of class labels (same length as data)
        p_value_threshold: P-value threshold (features with p > threshold are discarded)

    Returns:
        Tuple of (filtered_data, selected_feature_names)
    """
    if data.empty or len(data.columns) == 0:
        return data, []

    # Ensure all column names are strings
    data.columns = data.columns.astype(str)

    # Remove any NaN values in labels
    valid_mask = ~np.isnan(labels)
    if valid_mask.sum() < len(labels):
        print(
            f"  Warning: {len(labels) - valid_mask.sum()} samples with NaN labels removed for ANOVA"
        )
        data = data.loc[valid_mask]
        labels = labels[valid_mask]

    if len(data) == 0:
        return data, []

    # Fill NaN values in data with 0 (missing modalities)
    # This is necessary because ANOVA cannot handle NaN values
    n_nan = data.isna().sum().sum()
    if n_nan > 0:
        print(f"  Filling {n_nan} NaN values with 0 (missing modalities)")
        data = data.fillna(0)

    # Perform ANOVA F-test for each feature
    n_before = data.shape[1]

    # Get unique classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"  Warning: Only {len(unique_labels)} class(es) found, skipping ANOVA")
        return data, data.columns.tolist()

    # Calculate F-statistics and p-values
    f_scores, p_values = f_classif(data.values, labels)

    # Select features with p-value <= threshold
    selected_mask = p_values <= p_value_threshold
    selected_features = data.columns[selected_mask].tolist()

    n_after = len(selected_features)
    n_removed = n_before - n_after

    print(
        f"  After ANOVA filter (p <= {p_value_threshold}): {n_after} features (removed {n_removed})"
    )

    return data[selected_features], selected_features


###########################################################################
# MIRNA PREPROCESSING
###########################################################################


def load_mirna_data(
    filepath,
    remove_zeros=True,
    log_transform=True,
    normalize=True,
    remove_low_var=True,
    var_threshold=0.01,
):
    """
    Load and preprocess miRNA data.

    Based on data_exploration.ipynb:
    1. Load CSV with semicolon separator
    2. Set miRNA_ID as index
    3. Remove rows with all zeros
    4. Log2 transform
    5. Transpose (samples as rows, miRNAs as columns)
    6. Optionally normalize and remove low variance features

    Args:
        filepath: Path to miRNA CSV file
        remove_zeros: Whether to remove all-zero rows
        log_transform: Whether to apply log2(x+1) transform
        normalize: Whether to normalize to [0,1]
        remove_low_var: Whether to remove low variance features
        var_threshold: Variance threshold for feature selection

    Returns:
        Preprocessed DataFrame with samples as rows
    """
    print("Loading miRNA data...")

    # Load data
    mirna = pd.read_csv(filepath, sep=";")

    # Set miRNA_ID as index
    mirna = mirna.set_index("miRNA_ID")
    print(f"  Original shape: {mirna.shape[0]} miRNAs × {mirna.shape[1]} samples")

    # Remove all-zero rows
    if remove_zeros:
        zero_rows = (mirna == 0).all(axis=1)
        mirna = mirna[~zero_rows]
        print(
            f"  After removing zeros: {mirna.shape[0]} miRNAs × {mirna.shape[1]} samples"
        )

    # Log2 transform
    if log_transform:
        mirna = np.log2(mirna + 1)

    # Transpose: samples as rows, miRNAs as columns
    mirna = mirna.T

    # Standardize patient IDs
    mirna.index = [standardize_patient_id(idx) for idx in mirna.index]

    # Remove low variance features
    if remove_low_var:
        n_before = mirna.shape[1]
        mirna, _ = remove_low_variance_features(mirna, var_threshold)
        print(
            f"  After variance filter: {mirna.shape[1]} features (removed {n_before - mirna.shape[1]})"
        )

    # Normalize
    if normalize:
        mirna, _ = normalize_data(mirna, method="minmax")

    print(f"  Final shape: {mirna.shape[0]} samples × {mirna.shape[1]} features")

    return mirna


###########################################################################
# RNASEQ PREPROCESSING
###########################################################################


def load_rnaseq_data(
    filepath, remove_zeros=True, log_transform=True, var_threshold=0.02, normalize=True
):
    """
    Load and preprocess RNAseq data.

    Based on data_exploration.ipynb:
    1. Load CSV with semicolon separator
    2. Set gene_id as index
    3. Transpose (samples as rows)
    4. Remove all-zero columns
    5. Log2 transform
    6. Normalize and remove low variance features

    Args:
        filepath: Path to RNAseq CSV file
        remove_zeros: Whether to remove all-zero columns
        log_transform: Whether to apply log2(x+1) transform
        var_threshold: Variance threshold for feature selection

    Returns:
        Preprocessed DataFrame with samples as rows
    """
    print("Loading RNAseq data...")

    # Load data with chunking for large files
    rnaseq = pd.read_csv(filepath, sep=";")

    # Set gene_id as index and transpose
    rnaseq = rnaseq.set_index("gene_id")
    rnaseq = rnaseq.T
    print(f"  Original shape: {rnaseq.shape[0]} samples × {rnaseq.shape[1]} genes")

    # Remove all-zero columns
    if remove_zeros:
        non_zero_cols = (rnaseq != 0).any(axis=0)
        rnaseq = rnaseq.loc[:, non_zero_cols]
        print(f"  After removing zero columns: {rnaseq.shape[1]} genes")

    # Log2 transform
    if log_transform:
        rnaseq = np.log2(rnaseq + 1)

    # Standardize patient IDs
    rnaseq.index = [standardize_patient_id(idx) for idx in rnaseq.index]

    # Normalize (only if requested; otherwise caller handles it after train/test split)
    if normalize:
        rnaseq, _ = normalize_data(rnaseq, method="minmax")

    # Remove low variance features
    n_before = rnaseq.shape[1]
    rnaseq, _ = remove_low_variance_features(rnaseq, var_threshold)
    print(f"  After variance filter: {rnaseq.shape[1]} features")

    print(f"  Final shape: {rnaseq.shape[0]} samples × {rnaseq.shape[1]} features")

    return rnaseq


###########################################################################
# DNA METHYLATION PREPROCESSING
###########################################################################


def load_dna_methylation_data(
    filepath, impute_method="drop", completeness_threshold=0.80, n_neighbors=5, normalize=True
):
    """
    Load raw DNA methylation data from CpG-level file.

    Based on dna_methylation.ipynb workflow:
    1. Load CpG-level data (BRCA.DNAmethy_filtered.csv)
    2. Filter CpGs with ≥completeness_threshold data completeness
    3. Handle remaining NaN values (drop, mean, median, KNN, or iterative imputation)
    4. Transpose to get patients as rows

    Args:
        filepath: Path to DNA methylation CSV file (CpG × patients)
        impute_method: Method to handle NaN values. Options:
            - "drop": Drop CpGs with any remaining NaN values (default)
            - "mean": Impute NaN values with the mean of that CpG across patients
            - "median": Impute NaN values with the median of that CpG across patients
            - "knn": Impute based on k-nearest neighbors using patient similarity
            - "iterative": Iterative imputation (MICE algorithm)
        completeness_threshold: Minimum fraction of non-missing values per CpG (default: 0.80)
        n_neighbors: Number of neighbors for KNN imputation (default: 5)

    Returns:
        DataFrame with patients as rows, CpGs as columns
    """
    import time

    print("Loading DNA methylation data...")
    start_time = time.time()

    # Load data with progress
    print("  Reading CSV file...")
    dna_meth = pd.read_csv(filepath, sep=";")
    print(
        f"  Original shape: {dna_meth.shape[0]} CpGs × {dna_meth.shape[1] - 1} patients"
    )

    # Set CpG_ID as index
    dna_meth = dna_meth.set_index("CpG_ID")

    # Step 1: Filter CpGs with at least completeness_threshold data completeness
    completeness_per_cpg = dna_meth.notna().sum(axis=1) / dna_meth.shape[1]
    cpgs_to_keep = completeness_per_cpg >= completeness_threshold

    print(
        f"  CpG sites with ≥{completeness_threshold * 100:.0f}% data: {cpgs_to_keep.sum()} "
        f"(removed {(~cpgs_to_keep).sum()})"
    )

    dna_meth_filtered = dna_meth[cpgs_to_keep]

    # Step 2: Handle remaining NaN values
    n_nans_before = dna_meth_filtered.isna().sum().sum()

    if impute_method == "drop":
        # Drop CpGs with any remaining NaN
        dna_meth_clean = dna_meth_filtered.dropna(axis=0)
        print(
            f"  After dropping NaNs: {dna_meth_clean.shape[0]} CpGs (dropped {dna_meth_filtered.shape[0] - dna_meth_clean.shape[0]} CpGs)"
        )

    elif impute_method == "mean":
        # Impute with mean of each CpG across patients
        print(f"  Imputing {n_nans_before} NaN values with CpG-wise mean...")
        print("  Computing means...")
        cpg_means = dna_meth_filtered.mean(axis=1)
        print(
            "  Filling NaN values (this may take a few minutes for large datasets)..."
        )

        # More efficient imputation using numpy
        dna_meth_clean = dna_meth_filtered.copy()
        values = dna_meth_clean.values
        means_array = cpg_means.values

        # Fill NaNs row by row with progress bar
        for i in tqdm(range(len(values)), desc="  Imputing CpGs", ncols=80):
            mask = np.isnan(values[i])
            if mask.any():
                values[i][mask] = means_array[i]

        dna_meth_clean = pd.DataFrame(
            values, index=dna_meth_filtered.index, columns=dna_meth_filtered.columns
        )
        n_nans_after = dna_meth_clean.isna().sum().sum()
        print(
            f"  After mean imputation: {n_nans_after} NaNs remaining, {dna_meth_clean.shape[0]} CpGs"
        )

    elif impute_method == "median":
        # Impute with median of each CpG across patients
        print(f"  Imputing {n_nans_before} NaN values with CpG-wise median...")
        print("  Computing medians...")
        cpg_medians = dna_meth_filtered.median(axis=1)
        print(
            "  Filling NaN values (this may take a few minutes for large datasets)..."
        )

        # More efficient imputation using numpy
        dna_meth_clean = dna_meth_filtered.copy()
        values = dna_meth_clean.values
        medians_array = cpg_medians.values

        # Fill NaNs row by row with progress bar
        for i in tqdm(range(len(values)), desc="  Imputing CpGs", ncols=80):
            mask = np.isnan(values[i])
            if mask.any():
                values[i][mask] = medians_array[i]

        dna_meth_clean = pd.DataFrame(
            values, index=dna_meth_filtered.index, columns=dna_meth_filtered.columns
        )
        n_nans_after = dna_meth_clean.isna().sum().sum()
        print(
            f"  After median imputation: {n_nans_after} NaNs remaining, {dna_meth_clean.shape[0]} CpGs"
        )

    elif impute_method == "knn":
        # KNN-based imputation using patient similarity
        from sklearn.impute import KNNImputer

        print(f"  Imputing {n_nans_before} NaN values with KNN (k={n_neighbors})...")
        print(
            "  Computing patient similarities and imputing (this may take a few minutes)..."
        )

        # Transpose: patients as rows, CpGs as columns
        dna_meth_transposed = dna_meth_filtered.T

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        imputed_values = imputer.fit_transform(dna_meth_transposed.values)

        # Transpose back: CpGs as rows
        dna_meth_clean = pd.DataFrame(
            imputed_values.T,
            index=dna_meth_filtered.index,
            columns=dna_meth_filtered.columns,
        )

        n_nans_after = dna_meth_clean.isna().sum().sum()
        print(
            f"  After KNN imputation: {n_nans_after} NaNs remaining, {dna_meth_clean.shape[0]} CpGs"
        )

    elif impute_method == "iterative":
        # Iterative imputation (MICE algorithm)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        print(
            f"  Imputing {n_nans_before} NaN values with Iterative Imputation (MICE)..."
        )
        print("  This may take several minutes for large datasets...")

        # Transpose: patients as rows, CpGs as columns
        dna_meth_transposed = dna_meth_filtered.T

        # Apply Iterative imputation
        imputer = IterativeImputer(max_iter=10, random_state=42, verbose=1)
        imputed_values = imputer.fit_transform(dna_meth_transposed.values)

        # Transpose back: CpGs as rows
        dna_meth_clean = pd.DataFrame(
            imputed_values.T,
            index=dna_meth_filtered.index,
            columns=dna_meth_filtered.columns,
        )

        n_nans_after = dna_meth_clean.isna().sum().sum()
        print(
            f"  After iterative imputation: {n_nans_after} NaNs remaining, {dna_meth_clean.shape[0]} CpGs"
        )

    else:
        raise ValueError(
            f"Invalid impute_method: {impute_method}. Must be 'drop', 'mean', 'median', 'knn', or 'iterative'"
        )

    # Step 3: Transpose to get patients as rows
    dna_meth_clean = dna_meth_clean.T

    # Standardize patient IDs
    dna_meth_clean.index = [standardize_patient_id(idx) for idx in dna_meth_clean.index]

    # Step 4: MinMax normalization (only if requested; otherwise caller handles it after train/test split)
    if normalize:
        print("  Applying MinMax normalization...")
        dna_meth_clean, _ = normalize_data(dna_meth_clean, method="minmax")

    print(
        f"  Final shape: {dna_meth_clean.shape[0]} patients × {dna_meth_clean.shape[1]} CpGs"
    )

    elapsed_time = time.time() - start_time
    print(f"  Methylation data loaded in {elapsed_time:.2f} seconds")

    return dna_meth_clean


def apply_anova_to_methylation(
    meth_data, labels, p_value_threshold=0.05, use_fdr_correction=True
):
    """
    Apply Welch's ANOVA to select significant CpG sites.

    Welch's ANOVA is more robust than standard ANOVA as it does not assume
    equal variances between groups.

    Based on dna_methylation.ipynb:
    - Performs Welch's one-way ANOVA (does not assume equal variances)
    - Applies FDR correction (Benjamini-Hochberg)
    - Returns only significant CpGs

    Args:
        meth_data: DataFrame (patients × CpGs)
        labels: Array of survival group labels (0=short, 1=long)
        p_value_threshold: P-value threshold (default 0.05)
        use_fdr_correction: Whether to use FDR correction (default True)

    Returns:
        DataFrame with only significant CpGs, and list of selected CpG IDs
    """
    import time
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    print("  Applying Welch's ANOVA for feature selection...")
    start_time = time.time()

    # Ensure labels align with data
    if len(meth_data) != len(labels):
        raise ValueError(
            f"Data and labels size mismatch: {len(meth_data)} vs {len(labels)}"
        )

    # Get unique classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"  Warning: Only {len(unique_labels)} class(es) found, skipping ANOVA")
        return meth_data, meth_data.columns.tolist()

    # Perform Welch's ANOVA F-test for each CpG
    print(f"  Computing Welch's F-statistics for {len(meth_data.columns)} CpGs...")

    # Welch's ANOVA implementation (does not assume equal variances)
    # For 2 groups, this is equivalent to Welch's t-test
    p_values = []
    f_scores = []

    if len(unique_labels) == 2:
        # For 2 groups: use Welch's t-test (more efficient)
        group0 = meth_data[labels == unique_labels[0]].values
        group1 = meth_data[labels == unique_labels[1]].values

        for i in range(meth_data.shape[1]):
            # Welch's t-test (does not assume equal variances)
            t_stat, p_val = stats.ttest_ind(group0[:, i], group1[:, i], equal_var=False)
            # Convert t-statistic to F-statistic (F = t²)
            f_scores.append(t_stat**2)
            p_values.append(p_val)
    else:
        # For k>2 groups: compute Welch's ANOVA manually
        # Welch's ANOVA = weighted one-way ANOVA
        for i in range(meth_data.shape[1]):
            group_data = [
                meth_data[labels == label].values[:, i] for label in unique_labels
            ]

            # Compute group statistics
            k = len(group_data)
            n_i = np.array([len(g) for g in group_data])
            mean_i = np.array([np.mean(g) for g in group_data])
            var_i = np.array([np.var(g, ddof=1) for g in group_data])

            # Weights (inverse of variance)
            w_i = n_i / var_i

            # Weighted grand mean
            grand_mean = np.sum(w_i * mean_i) / np.sum(w_i)

            # Welch's F-statistic
            numerator = np.sum(w_i * (mean_i - grand_mean) ** 2) / (k - 1)

            # Correction factor lambda
            lambda_val = (
                3 * np.sum((1 - w_i / np.sum(w_i)) ** 2 / (n_i - 1)) / (k**2 - 1)
            )

            denominator = 1 + 2 * (k - 2) * lambda_val / (k**2 - 1)

            f_stat = numerator / denominator

            # Approximate p-value using F-distribution
            # Degrees of freedom for Welch's ANOVA
            df1 = k - 1
            df2 = 1 / (
                3 * np.sum((1 - w_i / np.sum(w_i)) ** 2 / (n_i - 1)) / (k**2 - 1)
            )

            p_val = 1 - stats.f.cdf(f_stat, df1, df2)

            f_scores.append(f_stat)
            p_values.append(p_val)

    p_values = np.array(p_values)
    f_scores = np.array(f_scores)

    print(f"  Welch's ANOVA complete!")

    # Apply FDR correction
    if use_fdr_correction:
        print(f"  Applying FDR correction (Benjamini-Hochberg)...")
        reject, p_values_corrected, _, _ = multipletests(
            p_values, alpha=p_value_threshold, method="fdr_bh"
        )

        # Select significant CpGs
        significant_cpgs = meth_data.columns[reject].tolist()

        print(f"  Number of CpG sites tested: {len(meth_data.columns)}")
        print(
            f"  Significant CpGs (p < {p_value_threshold}, uncorrected): {(p_values < p_value_threshold).sum()}"
        )
        print(f"  Significant CpGs (FDR corrected): {len(significant_cpgs)}")
    else:
        # Use uncorrected p-values
        significant_mask = p_values <= p_value_threshold
        significant_cpgs = meth_data.columns[significant_mask].tolist()

        print(f"  Number of CpG sites tested: {len(meth_data.columns)}")
        print(f"  Significant CpGs (p < {p_value_threshold}): {len(significant_cpgs)}")

    # Filter to significant CpGs
    meth_significant = meth_data[significant_cpgs]

    # Build F-score lookup for the significant CpGs (for optional ranking later)
    all_columns = meth_data.columns.tolist()
    f_scores_for_significant = np.array(
        [f_scores[all_columns.index(c)] for c in significant_cpgs]
    )

    elapsed_time = time.time() - start_time
    print(f"  ANOVA completed in {elapsed_time:.2f} seconds")

    return meth_significant, significant_cpgs, f_scores_for_significant


def load_dna_methylation_3d(data_dir, max_features=2000, normalize=True):
    """
    Load DNA methylation data as 3D matrix (samples × features × channels).
    Channels: [min, mean, max] methylation values.

    Args:
        data_dir: Directory containing methylation CSV files
        max_features: Maximum features per channel
        normalize: Whether to normalize

    Returns:
        3D numpy array (samples × features × 3)
    """
    print("Loading DNA methylation data (3D)...")

    files = {
        "min": os.path.join(data_dir, "BRCA.DNAmeth_min.csv"),
        "mean": os.path.join(data_dir, "BRCA.DNAmeth_mean.csv"),
        "max": os.path.join(data_dir, "BRCA.DNAmeth_max.csv"),
    }

    data_dict = {}
    common_features = None
    common_samples = None

    for name, filepath in files.items():
        if os.path.exists(filepath):
            print(f"  Loading {name} file...")
            try:
                df = pd.read_csv(filepath, sep=";")
                first_col = df.columns[0]
                df = df.set_index(first_col)
                df = df.T
                df.index = [standardize_patient_id(idx) for idx in df.index]

                data_dict[name] = df

                if common_features is None:
                    common_features = set(df.columns)
                    common_samples = set(df.index)
                else:
                    common_features &= set(df.columns)
                    common_samples &= set(df.index)
            except Exception as e:
                print(f"    Error loading {name}: {e}")

    if len(data_dict) < 3:
        print("  Warning: Not all methylation files loaded!")
        return None

    # Filter to common features and samples
    common_features = list(common_features)[:max_features]
    common_samples = sorted(list(common_samples))

    # Stack into 3D array
    arrays = []
    for name in ["min", "mean", "max"]:
        df = data_dict[name].loc[common_samples, common_features]
        if normalize:
            df, _ = normalize_data(df, method="minmax")
        arrays.append(df.values)

    # Shape: (samples, features, 3)
    meth_3d = np.stack(arrays, axis=-1)

    print(f"  Final 3D shape: {meth_3d.shape}")

    return meth_3d, common_samples, common_features


###########################################################################
# CLINICAL DATA / LABELS
###########################################################################


def load_clinical_labels(filepath, label_column="OS.survival_group"):
    """
    Load clinical data and extract binary labels from survival groups.

    Args:
        filepath: Path to clinical CSV file
        label_column: Column name for labels (default: 'OS.survival_group')

    Returns:
        DataFrame with patient IDs and binary labels (0=short, 1=long)
    """
    print("Loading clinical data...")

    clinical = pd.read_csv(filepath)

    # Get patient ID column (usually first column)
    id_col = clinical.columns[0]

    # Extract relevant columns
    labels = clinical[[id_col, label_column]].copy()
    labels.columns = ["patient_id", "survival_group"]

    # Standardize patient IDs
    labels["patient_id"] = labels["patient_id"].apply(standardize_patient_id)

    # Filter to only keep "short" and "long" survival groups
    # Discard "not considered" and any missing values
    print(f"  Total patients in clinical data: {len(labels)}")
    print(f"  Survival group distribution before filtering:")
    print(f"    {labels['survival_group'].value_counts().to_dict()}")

    # Keep only "short" and "long"
    labels = labels[labels["survival_group"].isin(["short", "long"])]

    # Convert to binary labels: short=0, long=1
    labels["label"] = (labels["survival_group"] == "long").astype(int)

    # Drop the survival_group column (keep only patient_id and label)
    labels = labels[["patient_id", "label"]]

    print(f"  After filtering (keeping only 'short' and 'long'): {len(labels)} samples")
    print(f"  Label distribution: {labels['label'].value_counts().to_dict()}")
    print(f"    (0 = short-term survival, 1 = long-term survival)")

    return labels


###########################################################################
# DATA ALIGNMENT (for preprocessing)
###########################################################################


def show_modality_overlap(*dataframes, labels_df, modality_names=None):
    """
    Show statistics about patient-modality overlap.

    Args:
        *dataframes: Variable number of pandas DataFrames with patient IDs as index
        labels_df: DataFrame with 'patient_id' and 'label' columns
        modality_names: List of modality names (e.g., ['miRNA', 'RNAseq', 'Methylation'])

    Returns:
        DataFrame showing overlap statistics
    """
    if modality_names is None:
        modality_names = [f"Modality_{i + 1}" for i in range(len(dataframes))]

    # Get all patient IDs from labels
    all_patients = set(labels_df["patient_id"].values)

    # Get patients for each modality
    modality_patients = {}
    for i, df in enumerate(dataframes):
        if df is not None:
            modality_patients[modality_names[i]] = set(df.index)
        else:
            modality_patients[modality_names[i]] = set()

    # Create overlap matrix
    modalities = list(modality_patients.keys())
    n_modalities = len(modalities)

    # Count patients with different combinations
    overlap_stats = []

    # All patients (union)
    all_modality_patients = set()
    for patients in modality_patients.values():
        all_modality_patients |= patients

    # Patients with labels
    patients_with_labels = len(all_patients)

    # Patients in each modality
    for mod in modalities:
        n_patients = len(modality_patients[mod])
        overlap_stats.append(
            {
                "Modality": mod,
                "Patients": n_patients,
                "With Labels": len(modality_patients[mod] & all_patients),
            }
        )

    # Patients with all modalities (intersection)
    if all(df is not None for df in dataframes):
        intersection = set.intersection(*[modality_patients[mod] for mod in modalities])
        overlap_stats.append(
            {
                "Modality": "All Modalities (Intersection)",
                "Patients": len(intersection),
                "With Labels": len(intersection & all_patients),
            }
        )

    # Patients with any modality (union)
    union = set.union(*[modality_patients[mod] for mod in modalities])
    overlap_stats.append(
        {
            "Modality": "Any Modality (Union)",
            "Patients": len(union),
            "With Labels": len(union & all_patients),
        }
    )

    # Create DataFrame
    stats_df = pd.DataFrame(overlap_stats)

    # Print table
    print("\n" + "=" * 60)
    print("PATIENT-MODALITY OVERLAP STATISTICS")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    print("=" * 60)

    # Show pairwise overlaps
    print("\nPairwise Modality Overlaps:")
    print("-" * 60)
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i + 1 :]:
            overlap = len(modality_patients[mod1] & modality_patients[mod2])
            print(f"  {mod1} ∩ {mod2}: {overlap} patients")

    return stats_df


def align_samples(*dataframes, labels_df, use_union=True):
    """
    Align samples across multiple omics datasets based on patient IDs.
    Uses union of all patients (not intersection) to include patients with missing modalities.

    Args:
        *dataframes: Variable number of pandas DataFrames with patient IDs as index
        labels_df: DataFrame with 'patient_id' and 'label' columns
        use_union: If True, use union of all patients. If False, use intersection.

    Returns:
        Tuple of aligned DataFrames, labels array, and aligned labels DataFrame
    """
    # Show overlap statistics
    modality_names = ["miRNA", "RNAseq", "Methylation"]
    show_modality_overlap(
        *dataframes,
        labels_df=labels_df,
        modality_names=modality_names[: len(dataframes)],
    )

    if use_union:
        # Use union: include all patients that appear in any modality or have labels
        all_samples = set(labels_df["patient_id"].values)

        for df in dataframes:
            if df is not None:
                all_samples |= set(df.index)

        all_samples = sorted(list(all_samples))
        print(
            f"\nUsing UNION: Found {len(all_samples)} total patients (including those with missing modalities)"
        )
    else:
        # Use intersection: only patients present in all modalities
        common_samples = set(labels_df["patient_id"].values)

        for df in dataframes:
            if df is not None:
                common_samples &= set(df.index)

        all_samples = sorted(list(common_samples))
        print(
            f"\nUsing INTERSECTION: Found {len(all_samples)} common samples across all modalities"
        )

    # Align all dataframes (reindex to include all patients, fill missing with NaN)
    aligned = []
    for df in dataframes:
        if df is not None:
            # Reindex to include all patients, missing values will be NaN
            aligned_df = df.reindex(all_samples)
            aligned.append(aligned_df)
        else:
            aligned.append(None)

    # Get aligned labels (reindex to include all patients)
    labels_aligned = labels_df.set_index("patient_id").reindex(all_samples)

    # Filter out patients without labels
    has_labels = ~labels_aligned["label"].isna()
    patients_with_labels = labels_aligned[has_labels].index.tolist()

    print(f"  Patients with labels: {has_labels.sum()}")
    print(f"  Patients without labels (discarded): {(~has_labels).sum()}")

    # Filter all dataframes to only include patients with labels
    aligned_filtered = []
    for df in aligned:
        if df is not None:
            aligned_filtered.append(df.loc[patients_with_labels])
        else:
            aligned_filtered.append(None)

    # Filter labels
    labels_aligned = labels_aligned.loc[patients_with_labels]
    labels_array = labels_aligned["label"].values

    return tuple(aligned_filtered) + (labels_array, labels_aligned)


###########################################################################
# SAVE PREPROCESSED DATA
###########################################################################


def save_preprocessed_data(
    data_dir,
    output_dir,
    test_size=0.2,
    max_mirna_features=None,
    max_rnaseq_features=None,
    max_meth_features=None,
    meth_impute_method="drop",
    meth_completeness_threshold=0.80,
    meth_knn_neighbors=5,
    meth_anova_p_threshold=0.02,
    meth_use_fdr=True,
    random_state=42,
):
    """
    Preprocess all modalities with correct train/test split workflow.

    KEY WORKFLOW (to avoid data leakage):
    1. Load clinical labels and split patients into train/test FIRST
    2. Load and preprocess each modality separately
    3. For each modality:
       - Apply basic preprocessing (log transform, normalization, etc.)
       - Split into train/test based on pre-defined patient IDs
       - Apply ANOVA on TRAINING set only
       - Use selected features from train on test set
    4. Save preprocessed train/test data

    Args:
        data_dir: Directory containing raw data files
        output_dir: Directory to save preprocessed train/test CSV files
        test_size: Fraction of data for testing (default 0.2 = 20%)
        max_mirna_features: Maximum features for miRNA (None = keep all)
        max_rnaseq_features: Maximum features for RNAseq (None = keep all)
        max_meth_features: Maximum features for methylation (None = keep all)
        meth_impute_method: Method to handle NaN values in methylation data.
            Options: "drop", "mean", "median", "knn", "iterative" (default: "drop")
        meth_completeness_threshold: Minimum fraction of non-missing values per CpG
            (default: 0.80 = 80%)
        meth_knn_neighbors: Number of neighbors for KNN imputation (default: 5)
        meth_anova_p_threshold: P-value threshold for ANOVA feature selection (default: 0.05)
        meth_use_fdr: Whether to use FDR (Benjamini-Hochberg) correction for methylation
            ANOVA (default: True). Set to False to use uncorrected p-values.
        random_state: Random seed for train/test split

    Returns:
        Dictionary with paths to saved files and metadata
    """
    print("=" * 80)
    print("PREPROCESSING MULTI-OMICS DATA (TRAIN/TEST SPLIT FIRST)")
    print("=" * 80)

    # ========================================================================
    # STEP 1: LOAD CLINICAL DATA AND SPLIT PATIENTS INTO TRAIN/TEST
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: LOAD CLINICAL DATA AND SPLIT PATIENTS")
    print("=" * 60)

    clinical_path = os.path.join(data_dir, "BRCA.clinical.csv")
    labels = load_clinical_labels(clinical_path)

    # Split patient IDs into train/test
    patient_ids = labels["patient_id"].values
    y = labels["label"].values

    train_patients, test_patients, y_train_global, y_test_global = train_test_split(
        patient_ids, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nGlobal train/test split:")
    print(f"  Total patients with labels: {len(patient_ids)}")
    print(
        f"  Training patients: {len(train_patients)} ({len(train_patients) / len(patient_ids) * 100:.1f}%)"
    )
    print(
        f"  Test patients: {len(test_patients)} ({len(test_patients) / len(patient_ids) * 100:.1f}%)"
    )
    print(f"  Train label distribution: {np.bincount(y_train_global.astype(int))}")
    print(f"  Test label distribution: {np.bincount(y_test_global.astype(int))}")

    # Convert to sets for easy filtering
    train_patients_set = set(train_patients)
    test_patients_set = set(test_patients)

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ========================================================================
    # STEP 2: PROCESS miRNA
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: PROCESS miRNA")
    print("=" * 60)

    mirna_path = os.path.join(data_dir, "BRCA.miRNA_RPM_tumor.csv")

    # Load and preprocess miRNA (basic preprocessing only; scaling applied after split)
    mirna = load_mirna_data(mirna_path, remove_low_var=False, normalize=False)

    if mirna is not None:
        # Filter to patients with labels
        mirna_patients = set(mirna.index) & set(patient_ids)
        mirna_filtered = mirna.loc[list(mirna_patients)]

        # Split into train/test
        mirna_train_patients = list(set(mirna_filtered.index) & train_patients_set)
        mirna_test_patients = list(set(mirna_filtered.index) & test_patients_set)

        print(f"  miRNA patients with labels: {len(mirna_patients)}")
        print(f"    In train set: {len(mirna_train_patients)}")
        print(f"    In test set: {len(mirna_test_patients)}")

        if len(mirna_train_patients) > 0:
            mirna_train = mirna_filtered.loc[mirna_train_patients]
            labels_mirna_train = labels[
                labels["patient_id"].isin(mirna_train_patients)
            ].set_index("patient_id")
            labels_mirna_train = labels_mirna_train.loc[mirna_train.index]
            y_mirna_train = labels_mirna_train["label"].values

            # Apply ANOVA on TRAINING set only
            print("  Applying ANOVA on TRAINING set only...")
            mirna_train_selected, selected_features = anova_feature_selection(
                mirna_train, y_mirna_train, p_value_threshold=0.05
            )

            # Apply same feature selection to TEST set
            if len(mirna_test_patients) > 0:
                mirna_test = mirna_filtered.loc[mirna_test_patients]
                mirna_test_selected = mirna_test[selected_features]
                labels_mirna_test = labels[
                    labels["patient_id"].isin(mirna_test_patients)
                ].set_index("patient_id")
                labels_mirna_test = labels_mirna_test.loc[mirna_test.index]
            else:
                mirna_test_selected = None
                labels_mirna_test = None

            # MinMax scaling: fit on train only, then apply to test
            print("  Applying MinMax scaling (fit on train only)...")
            scaler_mirna = MinMaxScaler()
            mirna_train_scaled = scaler_mirna.fit_transform(mirna_train_selected.values)
            mirna_train_selected = pd.DataFrame(
                mirna_train_scaled,
                index=mirna_train_selected.index,
                columns=mirna_train_selected.columns,
            )
            if mirna_test_selected is not None:
                mirna_test_scaled = scaler_mirna.transform(mirna_test_selected.values)
                mirna_test_selected = pd.DataFrame(
                    mirna_test_scaled,
                    index=mirna_test_selected.index,
                    columns=mirna_test_selected.columns,
                )

            # Save
            mirna_train_path = os.path.join(train_dir, "mirna.csv")
            mirna_train_selected.to_csv(mirna_train_path)
            labels_mirna_train_path = os.path.join(train_dir, "labels_mirna.csv")
            labels_mirna_train.reset_index().to_csv(
                labels_mirna_train_path, index=False
            )
            print(f"  Saved: {mirna_train_path} ({mirna_train_selected.shape})")

            if mirna_test_selected is not None:
                mirna_test_path = os.path.join(test_dir, "mirna.csv")
                mirna_test_selected.to_csv(mirna_test_path)
                labels_mirna_test_path = os.path.join(test_dir, "labels_mirna.csv")
                labels_mirna_test.reset_index().to_csv(
                    labels_mirna_test_path, index=False
                )
                print(f"  Saved: {mirna_test_path} ({mirna_test_selected.shape})")
    else:
        mirna_train_selected = None
        mirna_test_selected = None

    # ========================================================================
    # STEP 3: PROCESS RNAseq
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: PROCESS RNAseq")
    print("=" * 60)

    rnaseq_path = os.path.join(data_dir, "BRCA.RNA_seq_TPM.csv")

    # Load and preprocess RNAseq (scaling applied after split)
    rnaseq = load_rnaseq_data(rnaseq_path, var_threshold=0.02, normalize=False)

    if rnaseq is not None:
        # Filter to patients with labels
        rnaseq_patients = set(rnaseq.index) & set(patient_ids)
        rnaseq_filtered = rnaseq.loc[list(rnaseq_patients)]

        # Split into train/test
        rnaseq_train_patients = list(set(rnaseq_filtered.index) & train_patients_set)
        rnaseq_test_patients = list(set(rnaseq_filtered.index) & test_patients_set)

        print(f"  RNAseq patients with labels: {len(rnaseq_patients)}")
        print(f"    In train set: {len(rnaseq_train_patients)}")
        print(f"    In test set: {len(rnaseq_test_patients)}")

        if len(rnaseq_train_patients) > 0:
            rnaseq_train = rnaseq_filtered.loc[rnaseq_train_patients]
            labels_rnaseq_train = labels[
                labels["patient_id"].isin(rnaseq_train_patients)
            ].set_index("patient_id")
            labels_rnaseq_train = labels_rnaseq_train.loc[rnaseq_train.index]
            y_rnaseq_train = labels_rnaseq_train["label"].values

            # Apply ANOVA on TRAINING set only
            print("  Applying ANOVA on TRAINING set only...")
            rnaseq_train_selected, selected_features = anova_feature_selection(
                rnaseq_train, y_rnaseq_train, p_value_threshold=0.05
            )

            # Apply same feature selection to TEST set
            if len(rnaseq_test_patients) > 0:
                rnaseq_test = rnaseq_filtered.loc[rnaseq_test_patients]
                rnaseq_test_selected = rnaseq_test[selected_features]
                labels_rnaseq_test = labels[
                    labels["patient_id"].isin(rnaseq_test_patients)
                ].set_index("patient_id")
                labels_rnaseq_test = labels_rnaseq_test.loc[rnaseq_test.index]
            else:
                rnaseq_test_selected = None
                labels_rnaseq_test = None

            # MinMax scaling: fit on train only, then apply to test
            print("  Applying MinMax scaling (fit on train only)...")
            scaler_rnaseq = MinMaxScaler()
            rnaseq_train_scaled = scaler_rnaseq.fit_transform(rnaseq_train_selected.values)
            rnaseq_train_selected = pd.DataFrame(
                rnaseq_train_scaled,
                index=rnaseq_train_selected.index,
                columns=rnaseq_train_selected.columns,
            )
            if rnaseq_test_selected is not None:
                rnaseq_test_scaled = scaler_rnaseq.transform(rnaseq_test_selected.values)
                rnaseq_test_selected = pd.DataFrame(
                    rnaseq_test_scaled,
                    index=rnaseq_test_selected.index,
                    columns=rnaseq_test_selected.columns,
                )

            # Save
            rnaseq_train_path = os.path.join(train_dir, "rnaseq.csv")
            rnaseq_train_selected.to_csv(rnaseq_train_path)
            labels_rnaseq_train_path = os.path.join(train_dir, "labels_rnaseq.csv")
            labels_rnaseq_train.reset_index().to_csv(
                labels_rnaseq_train_path, index=False
            )
            print(f"  Saved: {rnaseq_train_path} ({rnaseq_train_selected.shape})")

            if rnaseq_test_selected is not None:
                rnaseq_test_path = os.path.join(test_dir, "rnaseq.csv")
                rnaseq_test_selected.to_csv(rnaseq_test_path)
                labels_rnaseq_test_path = os.path.join(test_dir, "labels_rnaseq.csv")
                labels_rnaseq_test.reset_index().to_csv(
                    labels_rnaseq_test_path, index=False
                )
                print(f"  Saved: {rnaseq_test_path} ({rnaseq_test_selected.shape})")
        else:
            rnaseq_train_selected = None
            rnaseq_test_selected = None

    # ========================================================================
    # STEP 4: PROCESS DNA METHYLATION (with ANOVA + FDR correction)
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: PROCESS DNA METHYLATION")
    print("=" * 60)

    meth_path = os.path.join(data_dir, "BRCA.DNAmethy_filtered.csv")

    if os.path.exists(meth_path):
        # Load DNA methylation data (CpG-level)
        meth = load_dna_methylation_data(
            meth_path,
            impute_method=meth_impute_method,
            completeness_threshold=meth_completeness_threshold,
            n_neighbors=meth_knn_neighbors,
            normalize=False,  # scaling applied after train/test split
        )

        # Filter to patients with labels
        meth_patients = set(meth.index) & set(patient_ids)
        meth_filtered = meth.loc[list(meth_patients)]

        # Split into train/test
        meth_train_patients = list(set(meth_filtered.index) & train_patients_set)
        meth_test_patients = list(set(meth_filtered.index) & test_patients_set)

        print(f"  Methylation patients with labels: {len(meth_patients)}")
        print(f"    In train set: {len(meth_train_patients)}")
        print(f"    In test set: {len(meth_test_patients)}")

        if len(meth_train_patients) > 0:
            meth_train = meth_filtered.loc[meth_train_patients]
            labels_meth_train = labels[
                labels["patient_id"].isin(meth_train_patients)
            ].set_index("patient_id")
            labels_meth_train = labels_meth_train.loc[meth_train.index]
            y_meth_train = labels_meth_train["label"].values

            # Apply ANOVA on TRAINING set only
            fdr_status = "with FDR correction" if meth_use_fdr else "without FDR correction"
            print(f"  Applying ANOVA {fdr_status} on TRAINING set only...")
            meth_train_selected, selected_cpgs, f_scores_sel = apply_anova_to_methylation(
                meth_train,
                y_meth_train,
                p_value_threshold=meth_anova_p_threshold,
                use_fdr_correction=meth_use_fdr,
            )

            if max_meth_features is not None and len(selected_cpgs) > max_meth_features:
                top_idx = np.argsort(f_scores_sel)[::-1][:max_meth_features]
                selected_cpgs = [selected_cpgs[i] for i in top_idx]
                meth_train_selected = meth_train_selected[selected_cpgs]
                print(f"  Capped to top {max_meth_features} features by F-score "
                      f"(from {len(f_scores_sel)} ANOVA-selected)")

            # Apply same feature selection to TEST set
            if len(meth_test_patients) > 0:
                meth_test = meth_filtered.loc[meth_test_patients]
                meth_test_selected = meth_test[selected_cpgs]
                labels_meth_test = labels[
                    labels["patient_id"].isin(meth_test_patients)
                ].set_index("patient_id")
                labels_meth_test = labels_meth_test.loc[meth_test.index]
            else:
                meth_test_selected = None
                labels_meth_test = None

            # MinMax scaling: fit on train only, then apply to test
            print("  Applying MinMax scaling (fit on train only)...")
            scaler_meth = MinMaxScaler()
            meth_train_scaled = scaler_meth.fit_transform(meth_train_selected.values)
            meth_train_selected = pd.DataFrame(
                meth_train_scaled,
                index=meth_train_selected.index,
                columns=meth_train_selected.columns,
            )
            if meth_test_selected is not None:
                meth_test_scaled = scaler_meth.transform(meth_test_selected.values)
                meth_test_selected = pd.DataFrame(
                    meth_test_scaled,
                    index=meth_test_selected.index,
                    columns=meth_test_selected.columns,
                )

            # Save
            meth_train_path = os.path.join(train_dir, "methylation.csv")
            meth_train_selected.to_csv(meth_train_path)
            labels_meth_train_path = os.path.join(train_dir, "labels_methylation.csv")
            labels_meth_train.reset_index().to_csv(labels_meth_train_path, index=False)
            print(f"  Saved: {meth_train_path} ({meth_train_selected.shape})")

            if meth_test_selected is not None:
                meth_test_path = os.path.join(test_dir, "methylation.csv")
                meth_test_selected.to_csv(meth_test_path)
                labels_meth_test_path = os.path.join(test_dir, "labels_methylation.csv")
                labels_meth_test.reset_index().to_csv(
                    labels_meth_test_path, index=False
                )
                print(f"  Saved: {meth_test_path} ({meth_test_selected.shape})")
        else:
            meth_train_selected = None
            meth_test_selected = None
    else:
        print(f"  Warning: {meth_path} not found!")
        meth_train_selected = None
        meth_test_selected = None

    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    metadata = {
        "n_total_patients": len(patient_ids),
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "test_size": test_size,
        "random_state": random_state,
        "n_mirna_features": mirna_train_selected.shape[1]
        if mirna_train_selected is not None
        else 0,
        "n_rnaseq_features": rnaseq_train_selected.shape[1]
        if rnaseq_train_selected is not None
        else 0,
        "n_meth_features": meth_train_selected.shape[1]
        if meth_train_selected is not None
        else 0,
        "meth_impute_method": meth_impute_method,
        "meth_completeness_threshold": meth_completeness_threshold,
        "meth_knn_neighbors": meth_knn_neighbors,
        "meth_anova_p_threshold": meth_anova_p_threshold,
        "meth_use_fdr": meth_use_fdr,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "note": "Train/test split done FIRST, then ANOVA applied on train only to avoid data leakage.",
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    return metadata


# Note: Multi-omics data integration functions (MultiOmicsDataset, prepare_data_loaders, etc.)
# have been moved to data_loader.py for better separation of concerns.
# Import them from data_loader if needed for backward compatibility.


###########################################################################
# MAIN EXECUTION
###########################################################################

if __name__ == "__main__":
    import sys
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess multi-omics BRCA data for KD-SVAE-VCDN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"),
        help="Directory to save preprocessed train/test CSV files",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (e.g., 0.2 = 20%%)",
    )
    parser.add_argument(
        "--meth-impute-method",
        type=str,
        default="knn",
        choices=["drop", "mean", "median", "knn", "iterative"],
        help="Method to handle NaN values in methylation data",
    )
    parser.add_argument(
        "--meth-completeness-threshold",
        type=float,
        default=0.80,
        help="Minimum fraction of non-missing values per CpG (lower = more features)",
    )
    parser.add_argument(
        "--meth-knn-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN imputation",
    )
    parser.add_argument(
        "--meth-anova-p-threshold",
        type=float,
        default=0.02,
        help="P-value threshold for ANOVA feature selection (higher = more features)",
    )
    parser.add_argument(
        "--max-meth-features",
        type=int,
        default=None,
        help="Keep only the top N methylation features by F-score after ANOVA (None = keep all)",
    )
    parser.add_argument(
        "--no-fdr",
        action="store_true",
        help="Disable FDR (Benjamini-Hochberg) correction for methylation ANOVA",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )

    args = parser.parse_args()

    # Resolve paths
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please provide the correct path to your data directory.")
        sys.exit(1)

    # Preprocess and save data
    metadata = save_preprocessed_data(
        data_dir=data_dir,
        output_dir=output_dir,
        test_size=args.test_size,
        max_mirna_features=None,
        max_rnaseq_features=None,
        max_meth_features=args.max_meth_features,
        meth_impute_method=args.meth_impute_method,
        meth_completeness_threshold=args.meth_completeness_threshold,
        meth_knn_neighbors=args.meth_knn_neighbors,
        meth_anova_p_threshold=args.meth_anova_p_threshold,
        meth_use_fdr=not args.no_fdr,
        random_state=args.random_state,
    )

    print("\n\nPreprocessing complete!")
    print(f"Train data saved to: {metadata['train_dir']}")
    print(f"Test data saved to: {metadata['test_dir']}")
