from typing import Union

import matplotlib

# matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from nilearn.glm import expression_to_contrast_vector
from vif_for_contrasts import est_contrast_vifs

from dd_kable_analysis.config_loader import Config
from dd_kable_analysis.tseries_model.contrast_model import (
    make_beta_series_constrast_set,
)
from dd_kable_analysis.tseries_model.design_matrix import (
    make_design_matrix,
)


def vif_bin(v: float) -> int:
    if v <= 5:
        return 0
    if v <= 10:
        return 1
    return 2


VIF_CMAP = ListedColormap(['lightgreen', 'orange', 'red'])


def make_design_qa_figure(
    cfg: Config, sub_id: str, run: Union[int, str], show: bool = False
) -> plt.Figure:
    """
    Generate QA figure for a subject/run showing:
      1. Design matrix
      2. Accept - Reject contrast
      3. Contrast VIFs

    Parameters
    ----------
    cfg : Config
        Analysis configuration object.
    sub_id : str
        Subject identifier.
    run : int or str
        Run number.
    show : bool, default False
        Whether to display the figure immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for further saving/processing.
    """
    # --- Generate design matrix and contrasts ---
    behav_data, events_data, desmat = make_design_matrix(cfg, sub_id, run)
    # Scale for the purposes of plotting
    desmat_scaled = desmat / np.sqrt((desmat**2).sum(axis=0)).replace(0, 1e-12)
    contrasts = make_beta_series_constrast_set(desmat, behav_data)
    vifs = est_contrast_vifs(desmat, contrasts)
    vif_matrix = pd.DataFrame(vifs, index=[0]).round(1)

    # --- Accept-Reject contrast ---
    accept_reject_contrast = expression_to_contrast_vector(
        contrasts['accept_minus_reject'], desmat.columns
    )
    contrast_df = pd.DataFrame(
        accept_reject_contrast.reshape(1, -1),
        columns=desmat.columns,
    ).round(2)

    fig_width = 18
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(fig_width, 10),
        gridspec_kw={'height_ratios': [8, 1.5, 1.5]},
    )
    fig.suptitle(f'Subject: {sub_id}   |   Run: {run}', fontsize=16, y=0.98)
    # --- Design matrix heatmap ---
    sns.heatmap(
        desmat_scaled,
        cmap='viridis',
        cbar=False,
        ax=axes[0],
        yticklabels=False,
        xticklabels=desmat.columns,
        vmin=0,
        vmax=1,
    )
    axes[0].set_title('Design Matrix')

    # --- Accept-reject contrast heatmap ---
    sns.heatmap(
        contrast_df,
        annot=contrast_df.values,
        fmt='g',
        cmap='coolwarm',
        center=0,
        cbar=False,
        ax=axes[1],
        xticklabels=desmat.columns,
        yticklabels=False,
        annot_kws={'fontsize': 6},
    )
    axes[1].set_title('Accept - Reject Contrast')

    # --- VIF heatmap with color mapping ---
    # sns.heatmap can't accept per-cell colors directly via cmap,
    # so we set the facecolors using the colors dataframe

    # --- VIF cell colors ---
    vif_bins = vif_matrix.apply(lambda col: col.map(vif_bin))

    ax = axes[2]
    sns.heatmap(
        vif_bins,
        annot=vif_matrix,
        fmt='g',
        cmap=VIF_CMAP,
        cbar=False,
        ax=ax,
        yticklabels=['cVIF'],
        xticklabels=vif_matrix.columns,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 6},
        vmin=0,
        vmax=2,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Contrast VIFs')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    return fig, vif_matrix
