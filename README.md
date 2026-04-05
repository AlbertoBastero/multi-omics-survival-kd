# KD-SVAE-VCDN: Knowledge Distillation with Supervised VAEs for Multi-Omics Cancer Survival Prediction

This project implements a hierarchical Knowledge Distillation (KD) framework for multi-omics breast cancer survival prediction, applied to the TCGA-BRCA dataset. It extends the VCDN framework of Ranjbari et al. with supervised Variational Autoencoders (SVAEs) and a 3-level distillation hierarchy.

## Task

Binary classification of breast cancer patient survival:
- **Short-term survival** (OS < 5 years)
- **Long-term survival** (OS ≥ 5 years)

Data: 323 TCGA-BRCA patients with three omics modalities — miRNA expression, bulk RNAseq, and DNA methylation (450k array).

## Architecture

The model follows a 3-level hierarchical Knowledge Distillation scheme:

```
Level 1 — Single-modality teachers (3 SVAEs)
    TE_1: miRNA        (~252 patients)
    TE_2: RNAseq       (~257 patients)
    TE_3: Methylation  (~184 patients)
         ↓ soft labels
Level 2 — Pairwise teachers (3 SVAEs)
    TE_12: miRNA + RNAseq       (~184 patients)
    TE_13: miRNA + Methylation  (~184 patients)
    TE_23: RNAseq + Methylation (~184 patients)
         ↓ soft labels
Level 3 — Student (SVAE + fusion)
    STU: all 3 modalities        (~182 patients, complete cases)
```

Each level is trained on the correct patient subset (those with data for all required modalities), as required by the Ranjbari et al. framework. Soft labels from teachers are generated on the exact patient subset used by the next level.

The student uses either **VCDN** (outer product fusion) or **concatenation-based** fusion of per-view latent representations.

## Results

5-fold cross-validation on TCGA-BRCA (metric: balanced accuracy):

| Configuration | Balanced Accuracy |
|---|---|
| **VCDN concat fusion** (best) | **0.8830 ± 0.0601** |
| Meth 2000 features + KL linear annealing | 0.8760 ± 0.0599 |
| KL annealing — none (no annealing) | 0.8474 ± 0.0537 |
| KD weights sweep — a=1, b=10, c=0.1 | 0.8346 ± 0.0660 |
| L1 regularization | 0.8186 ± 0.1654 |
| Baseline (standard VCDN) | 0.7490 ± 0.0201 |

**Key findings:**
- Concatenation-based fusion marginally outperforms VCDN outer product fusion
- Reducing methylation to top-2000 ANOVA features (vs. 33k) achieves comparable performance
- KD weights (a=1, b=10, c=0.1) are the best-performing configuration in the sweep
- KL annealing does not consistently improve performance — the bottleneck is VCDN capacity

## Data

Data is **not included** in this repository due to size. Download from:
- [TCGA-BRCA via GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
- miRNA, RNAseq (TPM), DNA methylation (450k array), and clinical survival data

Place raw files in `data/` and run preprocessing:

```bash
cd KD-SVAE-VCDN
python preprocess.py
```

Preprocessing applies ANOVA-based feature selection **on training data only** (no leakage), with train/test split stratified by survival class.

| Modality | Raw features | Selected |
|---|---|---|
| miRNA | 1,881 | 111 |
| RNAseq | 60,660 | 1,854 |
| DNA Methylation | 486,427 CpGs | 450 (or 2,000) |

## Setup

```bash
pip install -r KD-SVAE-VCDN/requirements.txt
```

GPU is recommended but not required. Tested with PyTorch ≥ 1.12.

## Usage

All training is handled by `KD-SVAE-VCDN/run_training.py`.

**5-fold cross-validation with KD (recommended):**
```bash
cd KD-SVAE-VCDN
python run_training.py --cross_validation --n_folds 5
```

**Best configuration (concat fusion):**
```bash
python run_training.py --cross_validation --n_folds 5 --fusion concat
```

**Disable knowledge distillation (baseline):**
```bash
python run_training.py --cross_validation --no_distillation
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--fusion` | `vcdn` | Fusion strategy: `vcdn` or `concat` |
| `--temperature` | `2.0` | Softmax temperature for soft labels |
| `--kd_a/b/c` | `1/10/0.1` | KD loss weights per teacher pair |
| `--kl_annealing` | `none` | KL schedule: `none`, `linear`, `cyclical` |
| `--regularization` | `none` | Weight regularization: `none`, `l1`, `l2`, `elastic` |
| `--te_epochs` | `30` | Training epochs for teachers |
| `--stu_epochs` | `50` | Training epochs for student |
| `--save_dir` | `./checkpoints` | Directory to save model weights |

**Hyperparameter sweeps:**
```bash
python sweep_kd_weights.py       # sweep KD loss weights
python sweep_temperature.py      # sweep softmax temperature
python compare_kl_annealing.py   # compare KL annealing strategies
```

## Repository Structure

```
KD-SVAE-VCDN/
├── config.py              # Model architecture hyperparameters
├── VAEs.py                # SVAE encoder/decoder implementations
├── KD.py                  # Knowledge distillation training logic (focal loss)
├── data_loader.py         # Per-modality data loading with KD-aware stratification
├── preprocess.py          # Raw data preprocessing pipeline
├── train_test.py          # Training, evaluation, and plotting functions
├── run_training.py        # Main training pipeline orchestrator
├── sweep_kd_weights.py    # KD weight hyperparameter sweep
├── sweep_temperature.py   # Temperature hyperparameter sweep
├── compare_kl_annealing.py # KL annealing strategy comparison
├── test_pipeline.py       # Pipeline validation utilities
└── requirements.txt       # Python dependencies
data/                      # Raw and preprocessed data (not tracked)
```

## Reference

Ranjbari, S., et al. "Multi-omics cancer survival prediction via hierarchical modality distillation." *Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)*, 2023.
