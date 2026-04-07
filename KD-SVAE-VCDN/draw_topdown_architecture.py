# -*- coding: utf-8 -*-
"""
Generates a presentation-ready diagram of the Top-Down KD architecture.
Run: python draw_topdown_architecture.py
Output: topdown_architecture.png / topdown_architecture.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
C_L1   = '#2E86AB'   # teal-blue  – L1 teacher
C_L2   = '#A23B72'   # purple     – L2 pairwise teachers
C_L3   = '#F18F01'   # amber      – L3 students
C_INT  = '#C73E1D'   # red-orange – integration
C_DATA = '#3B1F2B'   # dark       – input data
C_KD   = '#6DBF67'   # green      – KD soft-label arrows
C_FUSE = '#888888'   # grey       – latent-rep arrows
BG     = '#FAFAFA'

def box(ax, x, y, w, h, label, sublabel='', color='#2E86AB',
        fontsize=10, subfontsize=7.5, radius=0.012, alpha=0.92):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.025, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='white', alpha=0.88, zorder=4)

def arrow(ax, x0, y0, x1, y1, color='#444444', lw=1.5,
          style='->', label='', label_color=None, label_fs=7):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.012, my, label, fontsize=label_fs,
                color=label_color or color, ha='left', va='center', zorder=5)

def curved_arrow(ax, x0, y0, x1, y1, rad=0.25, color='#444', lw=1.5, label=''):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)
    if label:
        mx = (x0 + x1) / 2 + 0.04
        my = (y0 + y1) / 2
        ax.text(mx, my, label, fontsize=7, color=color,
                ha='left', va='center', zorder=5)

# ── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.965, 'Top-Down Knowledge Distillation Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#222')
ax.text(0.5, 0.940, 'Multi-Omics Survival Prediction  |  KD-SVAE-VCDN',
        ha='center', va='center', fontsize=10, color='#555')

# ── Level labels (left margin) ───────────────────────────────────────────────
for txt, yc, col in [
    ('INPUT',   0.858, C_DATA),
    ('LEVEL 1', 0.738, C_L1),
    ('LEVEL 2', 0.540, C_L2),
    ('LEVEL 3', 0.340, C_L3),
    ('FUSION',  0.145, C_INT),
]:
    ax.text(0.038, yc, txt, ha='center', va='center', fontsize=8.5,
            fontweight='bold', color=col,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=col,
                      edgecolor='none', alpha=0.12))

# ─────────────────────────────────────────────────────────────────────────────
# ROW 0 — input modalities
# ─────────────────────────────────────────────────────────────────────────────
xs_data = [0.25, 0.50, 0.75]
data_labels = ['miRNA\n(mod 1)', 'RNAseq\n(mod 2)', 'Methylation\n(mod 3)']
for x, lbl in zip(xs_data, data_labels):
    box(ax, x, 0.858, 0.14, 0.060, lbl, color=C_DATA, fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — L1 teacher (all 3 modalities)
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 0.50, 0.730, 0.46, 0.072,
    'L1 — All-Modality Teacher  (TopDownTeacherL1)',
    sublabel='3 × VAE (enc+dec)  +  VCDN fusion  →  soft labels (T-scaled)',
    color=C_L1, fontsize=10)

# arrows: data → L1
for x in xs_data:
    arrow(ax, x, 0.826, x, 0.770, color=C_DATA, lw=1.4)
# converge to L1 box centre
arrow(ax, 0.25, 0.770, 0.34, 0.766, color=C_DATA, lw=1.2)
arrow(ax, 0.75, 0.770, 0.66, 0.766, color=C_DATA, lw=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — L2 pairwise teachers
# ─────────────────────────────────────────────────────────────────────────────
xs_l2 = [0.22, 0.50, 0.78]
l2_labels  = ['L2 — Teacher [1+2]', 'L2 — Teacher [1+3]', 'L2 — Teacher [2+3]']
l2_sub     = ['miRNA + RNAseq',     'miRNA + Methyl.',    'RNAseq + Methyl.']
for x, lbl, sub in zip(xs_l2, l2_labels, l2_sub):
    box(ax, x, 0.538, 0.24, 0.068, lbl, sublabel=sub, color=C_L2, fontsize=8.5)

# KD arrows L1 → L2 (soft labels, dashed green)
for x in xs_l2:
    ax.annotate('', xy=(x, 0.574), xytext=(0.50, 0.694),
                arrowprops=dict(arrowstyle='->', color=C_KD, lw=1.5,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'), zorder=2)
ax.text(0.72, 0.640, 'soft labels\n(KD)', fontsize=7.5, color=C_KD,
        ha='left', va='center')

# data arrows to L2 (only relevant modalities)
# L2[1+2] ← mod1, mod2
for src_x, tgt_x in [(0.25, 0.22), (0.50, 0.22)]:
    ax.annotate('', xy=(tgt_x, 0.574), xytext=(src_x, 0.826),
                arrowprops=dict(arrowstyle='->', color=C_DATA, lw=1.0,
                                linestyle='dotted',
                                connectionstyle='arc3,rad=0.05'), zorder=1)
# L2[1+3] ← mod1, mod3
for src_x, tgt_x in [(0.25, 0.50), (0.75, 0.50)]:
    ax.annotate('', xy=(tgt_x, 0.574), xytext=(src_x, 0.826),
                arrowprops=dict(arrowstyle='->', color=C_DATA, lw=1.0,
                                linestyle='dotted',
                                connectionstyle='arc3,rad=0.0'), zorder=1)
# L2[2+3] ← mod2, mod3
for src_x, tgt_x in [(0.50, 0.78), (0.75, 0.78)]:
    ax.annotate('', xy=(tgt_x, 0.574), xytext=(src_x, 0.826),
                arrowprops=dict(arrowstyle='->', color=C_DATA, lw=1.0,
                                linestyle='dotted',
                                connectionstyle='arc3,rad=-0.05'), zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — L3 single-modality students
# ─────────────────────────────────────────────────────────────────────────────
xs_l3 = [0.22, 0.50, 0.78]
l3_labels = ['L3 — Student 1', 'L3 — Student 2', 'L3 — Student 3']
l3_sub    = ['miRNA only',     'RNAseq only',    'Methylation only']
for x, lbl, sub in zip(xs_l3, l3_labels, l3_sub):
    box(ax, x, 0.338, 0.24, 0.068, lbl, sublabel=sub, color=C_L3, fontsize=8.5)

# KD arrows L2 → L3
# Student 1 ← L2[1+2] and L2[1+3]
curved_arrow(ax, 0.22, 0.504, 0.22, 0.374, rad=-0.0,
             color=C_KD, lw=1.5, label='')
curved_arrow(ax, 0.50, 0.504, 0.22, 0.374, rad=0.25, color=C_KD, lw=1.5)
# Student 2 ← L2[1+2] and L2[2+3]
curved_arrow(ax, 0.22, 0.504, 0.50, 0.374, rad=-0.25, color=C_KD, lw=1.5)
curved_arrow(ax, 0.78, 0.504, 0.50, 0.374, rad=0.25, color=C_KD, lw=1.5)
# Student 3 ← L2[1+3] and L2[2+3]
curved_arrow(ax, 0.50, 0.504, 0.78, 0.374, rad=-0.25, color=C_KD, lw=1.5)
curved_arrow(ax, 0.78, 0.504, 0.78, 0.374, rad=0.0, color=C_KD, lw=1.5)

ax.text(0.87, 0.445, 'soft labels\n(KD)', fontsize=7.5, color=C_KD,
        ha='left', va='center')

# data arrows to L3
for x in xs_l3:
    ax.annotate('', xy=(x, 0.374), xytext=(x, 0.826),
                arrowprops=dict(arrowstyle='->', color=C_DATA, lw=1.0,
                                linestyle='dotted',
                                connectionstyle='arc3,rad=0.0'), zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Integration module
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 0.50, 0.145, 0.46, 0.072,
    'Integration Module  (VCDN / Concat fusion)',
    sublabel='Frozen L3 latent reps (μ, σ²)  →  final survival prediction',
    color=C_INT, fontsize=10)

# arrows L3 → Integration (latent representations)
for x in xs_l3:
    arrow(ax, x, 0.304, x, 0.200, color=C_FUSE, lw=1.5)
arrow(ax, 0.22, 0.200, 0.34, 0.181, color=C_FUSE, lw=1.2)
arrow(ax, 0.78, 0.200, 0.66, 0.181, color=C_FUSE, lw=1.2)

# Output arrow
arrow(ax, 0.50, 0.109, 0.50, 0.068, color=C_INT, lw=2.0)
ax.text(0.50, 0.050, 'Survival Prediction', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_INT)

# ─────────────────────────────────────────────────────────────────────────────
# LOSS annotations (right margin)
# ─────────────────────────────────────────────────────────────────────────────
loss_items = [
    (0.938, 0.730, C_L1,
     'Loss L1:\nBCE + β·KLD + 5·Focal'),
    (0.938, 0.538, C_L2,
     'Loss L2:\nBCE + β·KLD + 10·Focal\n+ 2·α·KL(p‖q_L1)'),
    (0.938, 0.338, C_L3,
     'Loss L3:\nBCE + β·KLD + 10·Focal\n+ 2·(a·KL₁ + b·KL₂)'),
    (0.938, 0.145, C_INT,
     'Loss INT:\nFocal only'),
]
for x, y, col, txt in loss_items:
    ax.text(x, y, txt, ha='center', va='center', fontsize=7,
            color=col, style='italic',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=col,
                      edgecolor='none', alpha=0.10))

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_DATA,  label='Raw input data'),
    mpatches.Patch(facecolor=C_L1,    label='L1 all-modality teacher'),
    mpatches.Patch(facecolor=C_L2,    label='L2 pairwise teachers × 3'),
    mpatches.Patch(facecolor=C_L3,    label='L3 single-modality students × 3'),
    mpatches.Patch(facecolor=C_INT,   label='Integration module'),
    mpatches.Patch(facecolor=C_KD,    label='KD soft-label flow'),
    mpatches.Patch(facecolor=C_FUSE,  label='Latent representation flow'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=7.5,
          framealpha=0.85, edgecolor='#ccc',
          bbox_to_anchor=(0.0, 0.0))

# ─────────────────────────────────────────────────────────────────────────────
# Divider lines between levels
# ─────────────────────────────────────────────────────────────────────────────
for y_div in [0.800, 0.630, 0.432, 0.225]:
    ax.axhline(y_div, xmin=0.07, xmax=0.93,
               color='#cccccc', lw=0.8, linestyle='--', zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
for fmt in ('png', 'pdf'):
    out = f'topdown_architecture.{fmt}'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f'Saved  {out}')
plt.close()
