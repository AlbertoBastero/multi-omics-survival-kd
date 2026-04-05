"""
Bar plot showing the number of patients in each OS.survival_group
for each modality (miRNA, DNA methylation, RNA seq).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load clinical data
clinical = pd.read_csv('data/BRCA.clinical.csv')
clinical_patients = set(clinical['bcr_patient_barcode'].values)

# Function to standardize patient IDs (convert dots to hyphens)
def standardize_id(patient_id):
    return patient_id.replace('.', '-')

# Load patient IDs from each modality (columns are patients)
# miRNA - uses dots in IDs
mirna_header = pd.read_csv('data/BRCA.miRNA_RPM_tumor.csv', sep=';', nrows=0)
mirna_patients = set(standardize_id(col) for col in mirna_header.columns[1:])

# RNAseq - uses hyphens in IDs
rnaseq_header = pd.read_csv('data/BRCA.RNA_seq_TPM.csv', sep=';', nrows=0)
rnaseq_patients = set(col for col in rnaseq_header.columns[1:])

# DNA Methylation - uses dots in IDs
methylation_header = pd.read_csv('data/BRCA.DNAmethy_filtered.csv', sep=';', nrows=0)
methylation_patients = set(standardize_id(col) for col in methylation_header.columns[1:])

# Define survival groups and modalities
survival_groups = ['not considered', 'short', 'long']
modalities = ['miRNA', 'DNA Methylation', 'RNA-seq']
modality_patients = [mirna_patients, methylation_patients, rnaseq_patients]

# Count patients per survival group for each modality
counts = {group: [] for group in survival_groups}

for modality_name, modality_set in zip(modalities, modality_patients):
    for group in survival_groups:
        # Get patients in this survival group
        group_patients = set(clinical[clinical['OS.survival_group'] == group]['bcr_patient_barcode'].values)
        # Count intersection with modality
        count = len(group_patients & modality_set)
        counts[group].append(count)

# Create the bar plot
x = np.arange(len(modalities))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Use shades of blue
colors = ['#87CEEB', '#4682B4', '#1E3A5F']  # Sky blue, Steel blue, Dark blue

bars = []
for i, (group, color) in enumerate(zip(survival_groups, colors)):
    bar = ax.bar(x + (i - 1) * width, counts[group], width, label=group, color=color, edgecolor='black', linewidth=0.5)
    bars.append(bar)

# Add value labels on bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize the plot
ax.set_xlabel('Modality', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Patient Counts per OS Survival Group by Modality\n(BRCA Dataset)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modalities, fontsize=11)
ax.legend(title='OS Survival Group', loc='upper right')
ax.set_ylim(0, max(max(counts[g]) for g in survival_groups) * 1.15)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('patients_per_modality_barplot.png', dpi=150, bbox_inches='tight')
print("Bar plot saved to 'patients_per_modality_barplot.png'")

# Print summary table
print("\n" + "=" * 60)
print("Patient Counts Summary")
print("=" * 60)
print(f"{'Modality':<20} {'Not Considered':>15} {'Short':>10} {'Long':>10} {'Total':>10}")
print("-" * 60)
for i, modality in enumerate(modalities):
    nc = counts['not considered'][i]
    short = counts['short'][i]
    long = counts['long'][i]
    total = nc + short + long
    print(f"{modality:<20} {nc:>15} {short:>10} {long:>10} {total:>10}")
print("=" * 60)
