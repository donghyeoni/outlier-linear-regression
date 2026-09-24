"""Matplotlib helpers for convergence plots and per-dataset strip plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .optimizers import OPTIMIZERS


def plot_optimizer_convergence(histories, metric, batch, init, *,
                               optimizer_list=OPTIMIZERS, ylabel=None,
                               title=None, save_path=None, show=False):
    """Plot per-epoch ``metric`` curves for each optimizer at a fixed
    ``(batch, init)`` configuration.

    Parameters
    ----------
    histories : dict
        Output of :func:`outlier_regression.train.run_experiment`, keyed by
        ``(optimizer, batch, init)``.
    metric : {"estimation", "weight"}
        Which recorded history to plot.
    batch, init : str
        The batch scheme and init scheme to slice on.
    ylabel, title : str, optional
    save_path : str, optional
        If given, the figure is written to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(5, 3))
    for opt in optimizer_list:
        key = (opt, batch, init)
        if key in histories:
            plt.plot(histories[key][metric], label=opt)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel or ("Estimation Error" if metric == "estimation"
                          else "Weight Error (L2)"))
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_curves(curves, *, xlabel="Epoch", ylabel="Value", title=None,
                save_path=None, show=False):
    """Plot a dict of ``label -> sequence`` as overlaid line curves.

    Useful for the learning-rate sweep in the outlier experiment.
    """
    fig = plt.figure(figsize=(5, 3))
    for label, series in curves.items():
        plt.plot(series, label=str(label))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_weight_error_strip(series, *, save_path, title, point_size=28,
                            jitter=0.12, edge_width=1.0, alpha=None):
    """Strip plot of per-dataset weight errors, one column per method.

    Parameters
    ----------
    series : list of (label, values, color)
        One entry per method, drawn left to right. Each column shows every
        value as a jittered point, a black bar at the median and the median
        as text.
    save_path : str
        Where the figure is written (dpi 150).
    title : str
    point_size, jitter, edge_width, alpha
        Marker area, half-width of the horizontal jitter, white marker edge
        width and marker opacity (None = opaque).

    The jitter uses ``numpy.random.default_rng(0)``, drawn column by column,
    so the figure is reproducible.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (_, values, color) in enumerate(series):
        v = np.asarray(values)
        x = i + rng.uniform(-jitter, jitter, size=len(v))
        ax.scatter(x, v, s=point_size, color=color, edgecolor="white",
                   linewidth=edge_width, alpha=alpha, zorder=3)
        med = np.median(v)
        ax.hlines(med, i - 0.25, i + 0.25, color="#0b0b0b", lw=2, zorder=4)
        ax.annotate(f"{med:.3f}", (i + 0.28, med), va="center", fontsize=8,
                    color="#52514e")
    ax.set_yscale("log")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([label for label, _, _ in series])
    ax.set_xlim(-0.5, len(series) - 0.3)
    ax.set_ylabel("weight error (log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", which="both", color="#e5e5e2", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
