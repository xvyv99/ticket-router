"""Shared matplotlib/seaborn style configuration and figure save helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def set_academic_style(font_size: int = 11) -> None:
    """Apply consistent academic/paper-quality style across all figures.

    Sets sans-serif font, 300 DPI, removes top/right spines, tight bbox,
    and configures seaborn for "paper" context.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_context("paper", font_scale=1.1)
    sns.set_palette("muted")


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    fmt: str = "png",
    close: bool = True,
) -> None:
    """Save a matplotlib figure to disk and optionally close it."""
    fig.savefig(path, format=fmt)
    if close:
        plt.close(fig)
