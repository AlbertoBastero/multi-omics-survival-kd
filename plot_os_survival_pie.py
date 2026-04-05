"""
Simple script to plot a pie chart showing the distribution of OS.survival_group
in the BRCA clinical data.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the clinical data
clinical_data = pd.read_csv('data/BRCA.clinical.csv')

# Count the occurrences of each OS.survival_group
survival_counts = clinical_data['OS.survival_group'].value_counts()

# Define the order and labels
categories = ['not considered', 'short', 'long']
counts = [survival_counts.get(cat, 0) for cat in categories]
labels = [f'{cat}\n(n={count})' for cat, count in zip(categories, counts)]

# Define shades of green (light to dark)
green_shades = ['#90EE90', '#3CB371', '#228B22']  # Light green, Medium sea green, Forest green

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    counts,
    labels=labels,
    colors=green_shades,
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.02, 0.02, 0.02),  # Slight separation between slices
    shadow=True
)

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

# Style the labels
for text in texts:
    text.set_fontsize(11)

ax.set_title('Distribution of OS Survival Groups\n(BRCA Clinical Data)', 
             fontsize=14, fontweight='bold', pad=20)

# Add a legend
ax.legend(wedges, categories, 
          title="Survival Groups",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.savefig('os_survival_pie_chart.png', dpi=150, bbox_inches='tight')
print("\nPie chart saved to 'os_survival_pie_chart.png'")

print(f"\nSurvival Group Counts:")
print("-" * 30)
for cat, count in zip(categories, counts):
    print(f"  {cat}: {count}")
print("-" * 30)
print(f"  Total: {sum(counts)}")
