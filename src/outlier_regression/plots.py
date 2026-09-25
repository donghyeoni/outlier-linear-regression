"""Matplotlib helpers: learning curves and per-dataset strip plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(panels, *, save_path, ylabel="weight error", log_y=True):
    """Side-by-side panels of overlaid curves.

    ``panels`` is a list of ``(title, {label: sequence})``. The x axis is the
    iteration (1-based).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2),
                             sharey=True, squeeze=False)
    for ax, (title, curves) in zip(axes[0], panels):
        for label, seq in curves.items():
            ax.plot(np.arange(1, len(seq) + 1), seq, label=label, lw=1.4)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", color="#e5e5e2", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_strip(series, *, save_path, title, ylabel="weight error (log scale)",
               figsize=(6, 3.4)):
    """Strip plot of per-dataset values, one column per method.

    ``series`` is a list of ``(label, values, color)``. Each column shows
    every value as a jittered point and a black bar with the median, printed
    to 4 decimals. The jitter uses ``numpy.random.default_rng(0)``.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=figsize)
    for i, (_, values, color) in enumerate(series):
        v = np.asarray(values)
        x = i + rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(x, v, s=22, color=color, edgecolor="white", linewidth=0.8,
                   zorder=3)
        med = float(np.median(v))
        ax.hlines(med, i - 0.25, i + 0.25, color="#0b0b0b", lw=2, zorder=4)
        ax.annotate(f"{med:.4f}", (i + 0.27, med), va="center", fontsize=7.5,
                    color="#52514e")
    ax.set_yscale("log")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=8)
    ax.set_xlim(-0.5, len(series) - 0.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", which="both", color="#e5e5e2", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
