from __future__ import annotations

"""
High-level decoding routines.

This module contains:
- subject-level atlas ROI decoding (nested group CV ridge regression)
- helper to paint ROI-level scores back into an atlas image for visualization
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dd_kable_analysis.mvpa.data import build_subject_behav_bold_df
from dd_kable_analysis.mvpa.features import (
    filter_voxels_runaware,
    prepare_subject_for_atlas_mvpa,
)
from dd_kable_analysis.mvpa.models import nested_groupcv_ridge_predict


def decode_subject_atlas_rois(
    cfg: Any,
    sub_id: str,
    *,
    atlas_img: str | Path | Any,
    y_col: str = 'amount',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    min_voxels_per_roi: int = 50,
    small_thr: float = 1e-4,
    max_small_frac: float = 0.05,
    require_all_runs: bool = True,
    alphas: np.ndarray | None = None,
    verbose: bool = True,
    return_trialwise: bool = False,
    trialwise_rois: set[int] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decode a behavioral variable from beta-series patterns within atlas ROIs (per subject).

    For each ROI label in `atlas_img`, this function:
      1) extracts trial × voxel data from that ROI
      2) applies run-aware voxel QC
      3) runs nested group CV ridge regression (typically leave-one-run-out)
      4) stores out-of-sample decoding metrics (r, Fisher-z(r), fold-safe R^2_cv, etc.)

    Parameters
    ----------
    cfg
        Analysis config object (used by build_subject_behav_bold_df and to locate masks).
    sub_id
        Subject ID string.
    atlas_img
        3D integer label atlas (path or Nifti1Image). Example: Schaefer, Harvard–Oxford.
        Labels must be integers; 0 is treated as background by ROI mapping code.
    y_col
        Column name in the behavioral/design table to decode (e.g., "amount", "delay").
    beta_col
        Column name containing beta NIfTI file paths (usually "beta_file").
    group_col
        Column name defining CV grouping (usually "run").
    min_voxels_per_roi
        Skip ROIs with fewer than this many voxels (pre- or post-QC).
    small_thr, max_small_frac, require_all_runs
        Parameters for run-aware voxel QC (`filter_voxels_runaware`).
    alphas
        Ridge alpha grid for nested CV. If None, uses model default.
    verbose
        If True, prints trial/run QA information during subject table creation and prep.
    return_trialwise
        If True, also return a trialwise DataFrame with out-of-sample predictions ŷ.
    trialwise_rois
        If return_trialwise=True, restrict trialwise output to these ROI labels.
        If None and return_trialwise=True, a ValueError is raised to prevent huge outputs.

    Returns
    -------
    roi_summary_df
        DataFrame with one row per ROI that passes voxel thresholds. Columns include:
        sub_id, roi_label, n_trials, n_runs, n_vox_preQC, n_vox_postQC,
        r, fisher_z, r2_cv, rmse, mean_alpha.
    (roi_summary_df, trialwise_df)
        If return_trialwise=True, also returns trialwise_df with columns like:
        sub_id, roi_label, run, [trial_type/delay/choseAccept if present], y, yhat_oos.

    Notes
    -----
    All decoding metrics are computed from out-of-sample predictions produced by the
    outer CV loop.
    """
    if return_trialwise and trialwise_rois is None:
        raise ValueError(
            'return_trialwise=True with trialwise_rois=None will generate a huge '
            'trialwise table (trials × all ROIs). Pass trialwise_rois (set of ints).'
        )

    # 1) build df of behavior + beta files (one row per trial)
    out = build_subject_behav_bold_df(cfg, sub_id=sub_id, verbose=verbose)
    behav_bold_df = out.behav_bold_df

    # 2) global extraction + ROI mapping
    prep = prepare_subject_for_atlas_mvpa(
        behav_bold_df,
        atlas_img=atlas_img,
        cfg=cfg,
        y_col=y_col,
        beta_col=beta_col,
        group_col=group_col,
        standardize_X=False,
        verbose=verbose,
    )

    rows: list[dict[str, Any]] = []
    trialwise_parts: list[pd.DataFrame] = []

    for roi_label, cols in prep.roi_to_cols.items():
        if cols.size == 0:
            continue

        X_roi = prep.X_all[:, cols]
        n_vox_pre = int(X_roi.shape[1])
        if n_vox_pre < min_voxels_per_roi:
            continue

        # subject-specific voxel QC (run-aware)
        try:
            X_roi_f, _qc = filter_voxels_runaware(
                X_roi,
                prep.groups,
                small_thr=small_thr,
                max_small_frac=max_small_frac,
                require_all_runs=require_all_runs,
                verbose=False,
            )
        except RuntimeError:
            continue

        n_vox_post = int(X_roi_f.shape[1])
        if n_vox_post < min_voxels_per_roi:
            continue

        # decode (OOS predictions across all trials)
        yhat, info = nested_groupcv_ridge_predict(
            X_roi_f, prep.y, prep.groups, alphas=alphas, verbose=False
        )

        # ---- trialwise output (optional) ----
        if return_trialwise and (
            trialwise_rois is None or int(roi_label) in trialwise_rois
        ):
            df_tw = prep.df_used.copy().reset_index(drop=True)

            if 'run' in df_tw.columns:
                df_tw['run'] = df_tw['run'].astype(str)

            df_tw['sub_id'] = str(sub_id)
            df_tw['roi_label'] = int(roi_label)
            df_tw['y'] = prep.y
            df_tw['yhat_oos'] = yhat

            extra_cols = [
                c
                for c in ['trial_type', 'Delay', 'amount', 'choseAccept']
                if c in df_tw.columns
            ]
            cols_tw = ['sub_id', 'roi_label', 'run'] + extra_cols + ['y', 'yhat_oos']
            cols_tw = [c for c in cols_tw if c in df_tw.columns]
            trialwise_parts.append(df_tw[cols_tw])

        # ---- summary row ----
        r = float(info['r'])
        rows.append(
            dict(
                sub_id=str(sub_id),
                roi_label=int(roi_label),
                n_trials=int(len(prep.y)),
                n_runs=int(len(np.unique(prep.groups))),
                n_vox_preQC=n_vox_pre,
                n_vox_postQC=n_vox_post,
                r=r,
                r2_cv=float(info['r2_cv']),
                fisher_z=float(np.arctanh(np.clip(r, -0.999999, 0.999999))),
                rmse=float(info['rmse']),
                mean_alpha=float(np.mean(info['chosen_alphas']))
                if len(info['chosen_alphas'])
                else np.nan,
            )
        )

    roi_summary_df = (
        pd.DataFrame(rows).sort_values(['roi_label']).reset_index(drop=True)
    )

    if return_trialwise:
        trialwise_df = (
            pd.concat(trialwise_parts, ignore_index=True)
            if len(trialwise_parts)
            else pd.DataFrame()
        )
        return roi_summary_df, trialwise_df

    return roi_summary_df


def roi_scores_to_atlas_image(
    roi_summary_df: pd.DataFrame,
    atlas_img: str | Path | Any,
    *,
    score_col: str = 'r2_cv',
    background_value: float = 0.0,
    reference_img: str | Path | Any | None = None,
):
    """
    Paint ROI-level scores into a voxelwise image on the atlas grid.

    Parameters
    ----------
    roi_summary_df
        DataFrame with columns ['roi_label', score_col].
    atlas_img
        Label atlas image (path or Nifti1Image).
    score_col
        Column from roi_summary_df to paint into parcels (e.g., 'r2_cv', 'fisher_z').
    background_value
        Value for voxels where atlas label == 0.
    reference_img
        Optional reference image to resample the output onto (e.g., a beta image).

    Returns
    -------
    score_img
        nib.Nifti1Image with voxelwise values assigned per parcel.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img

    atlas_img = (
        nib.load(str(atlas_img)) if not hasattr(atlas_img, 'get_fdata') else atlas_img
    )
    atlas_data = atlas_img.get_fdata().astype(int)

    score_map = dict(
        zip(
            roi_summary_df['roi_label'].astype(int).to_numpy(),
            roi_summary_df[score_col].to_numpy(),
        )
    )

    out = np.full(atlas_data.shape, background_value, dtype=np.float32)
    for lab, val in score_map.items():
        out[atlas_data == lab] = np.float32(val)

    score_img = nib.Nifti1Image(out, affine=atlas_img.affine)

    if reference_img is not None:
        ref = (
            nib.load(str(reference_img))
            if not hasattr(reference_img, 'get_fdata')
            else reference_img
        )
        score_img = resample_to_img(
            score_img,
            ref,
            interpolation='continuous',
            force_resample=True,
            copy_header=True,
        )

    return score_img
