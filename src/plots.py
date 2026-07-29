"""Matplotlib helpers for convergence plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .optimizers import OPTIMIZERS


def plot_optimizer_convergence(histories, metric, batch, init, *,
                               optimizer_list=OPTIMIZERS, ylabel=None,
                               title=None, save_path=None, show=False):
    """Plot per-epoch ``metric`` curves for each optimizer at a fixed
    ``(batch, init)`` configuration.

    Parameters
    ----------
    histories : dict
        Output of :func:`src.train.run_experiment`, keyed by
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
