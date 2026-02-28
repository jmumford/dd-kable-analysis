from __future__ import annotations

"""
Feature extraction and ROI mapping utilities for atlas-based MVPA.

This module handles:
- run-aware voxel QC (drop voxels that are near-zero too often within runs)
- extracting a global trial × voxel feature matrix (X_all) from beta-series images
  using a group mask
- mapping a label atlas (e.g., Schaefer, Harvard–Oxford) into column indices
  of X_all via a shared NiftiMasker voxel ordering
- packaging everything needed for downstream ROI-wise decoding
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker


@dataclass
class VoxelFilterInfo:
    """Metadata returned by run-aware voxel filtering."""

    n_vox_in: int
    n_vox_out: int
    keep_mask: np.ndarray  # shape: (n_vox_in,)
    frac_small_by_run: dict[str, np.ndarray]  # run -> (n_vox_in,) fraction near-zero
    params: dict[str, Any]


def filter_voxels_runaware(
    X: np.ndarray,
    groups: np.ndarray,
    *,
    small_thr: float = 1e-4,
    max_small_frac: float = 0.05,
    require_all_runs: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, VoxelFilterInfo]:
    """
    Run-aware voxel QC filter.

    Drops voxels that are "near zero" too often within any run. This is intended to
    remove voxels that are effectively absent / zeroed out due to dropout or masking.

    Parameters
    ----------
    X
        Array of shape (n_trials, n_voxels) for a single ROI.
    groups
        Run labels for each trial. Shape (n_trials,). Will be cast to str.
    small_thr
        Absolute value threshold below which a voxel value counts as "small".
    max_small_frac
        Maximum allowed fraction of trials in a run for which |X| < small_thr.
        Voxels exceeding this fraction are dropped.
    require_all_runs
        If True, a voxel must pass the criterion in every run.
        If False, a voxel may fail in at most 1 run.
    verbose
        Currently unused (placeholder for debugging).

    Returns
    -------
    Xf
        Filtered X with shape (n_trials, n_vox_out).
    info
        VoxelFilterInfo with keep mask and per-run small fractions.

    Raises
    ------
    RuntimeError
        If no voxels remain after filtering.
    """
    X = np.asarray(X)
    groups = np.asarray(groups).astype(str)

    uniq_runs = np.unique(groups)
    n_vox = X.shape[1]
    frac_small_by_run: dict[str, np.ndarray] = {}
    good_by_run = []

    for r in uniq_runs:
        m = groups == r
        frac_small = np.mean(np.abs(X[m, :]) < small_thr, axis=0)
        frac_small_by_run[str(r)] = frac_small
        good_by_run.append(frac_small <= max_small_frac)

    good_by_run = np.stack(good_by_run, axis=0)

    keep = (
        np.all(good_by_run, axis=0)
        if require_all_runs
        else (np.sum(good_by_run, axis=0) >= (len(uniq_runs) - 1))
    )

    Xf = X[:, keep]
    info = VoxelFilterInfo(
        n_vox_in=int(n_vox),
        n_vox_out=int(Xf.shape[1]),
        keep_mask=keep,
        frac_small_by_run=frac_small_by_run,
        params=dict(
            small_thr=float(small_thr),
            max_small_frac=float(max_small_frac),
            require_all_runs=bool(require_all_runs),
        ),
    )

    if Xf.shape[1] == 0:
        raise RuntimeError('0 voxels remain after run-aware filtering.')

    return Xf, info


@dataclass
class SubjectPreparedMVPA:
    """
    Output of per-subject feature preparation for atlas-based decoding.
    """

    df_used: pd.DataFrame
    X_all: np.ndarray  # (n_trials, n_vox_global)
    y: np.ndarray  # (n_trials,)
    groups: np.ndarray  # (n_trials,) run labels (strings)
    ref_img: nib.Nifti1Image
    masker_global: NiftiMasker
    atlas_labels_vec: np.ndarray  # (n_vox_global,) integer atlas label per voxel/column
    roi_to_cols: dict[int, np.ndarray]  # roi_label -> column indices into X_all


def extract_subject_global_Xy_groups(
    behav_bold_df: pd.DataFrame,
    *,
    group_mask_file: str | Path,
    y_col: str = 'amount',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    standardize_X: bool = False,
    verbose: bool = False,
) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image, NiftiMasker
]:
    """
    Extract global X (trial × voxel) matrix using a group mask, and return y and run groups.

    This loads all beta-series NIfTIs listed in `behav_bold_df[beta_col]` and applies
    `group_mask_file` (resampling the mask only if needed) to obtain a consistent voxel
    ordering across trials.

    Parameters
    ----------
    behav_bold_df
        Trial table with at least columns [y_col, beta_col, group_col].
    group_mask_file
        Path to a 3D binary mask image in the same space as the beta images.
    y_col
        Column name in behav_bold_df for target y.
    beta_col
        Column name containing beta NIfTI file paths (one per trial).
    group_col
        Column name for grouping variable used for CV (e.g., run).
    standardize_X
        If True, have NiftiMasker standardize features (usually False because you
        standardize within CV folds in the model pipeline).
    verbose
        If True, print basic shapes and per-run trial counts.

    Returns
    -------
    df_used
        Cleaned trial DataFrame used for extraction (rows correspond to X rows).
    X_all
        Array shape (n_trials, n_vox_global).
    y
        Array shape (n_trials,).
    groups
        Run labels as strings, shape (n_trials,).
    ref_img
        First beta image loaded (used as reference grid).
    masker_global
        Fitted NiftiMasker used for extraction (voxel ordering consistent with X_all).
    """
    needed = {y_col, beta_col, group_col}
    missing = needed - set(behav_bold_df.columns)
    if missing:
        raise ValueError(f'Missing columns in behav_bold_df: {missing}')

    df_used = (
        behav_bold_df.dropna(subset=[y_col, beta_col, group_col])
        .copy()
        .reset_index(drop=True)
    )
    if df_used.empty:
        raise ValueError('behav_bold_df empty after dropping NAs.')

    beta_files = df_used[beta_col].astype(str).tolist()
    ref_img = nib.load(beta_files[0])

    group_mask_img = nib.load(str(group_mask_file))
    if group_mask_img.shape[:3] == ref_img.shape[:3] and np.allclose(
        group_mask_img.affine, ref_img.affine
    ):
        group_mask_rs = group_mask_img
    else:
        group_mask_rs = resample_to_img(
            group_mask_img,
            ref_img,
            interpolation='nearest',
            force_resample=False,
            copy_header=True,
        )

    masker_global = NiftiMasker(mask_img=group_mask_rs, standardize=standardize_X)
    masker_global.fit()
    X_all = masker_global.transform(beta_files)  # (n_trials, n_vox_global)

    y = df_used[y_col].to_numpy(dtype=float)
    groups = df_used[group_col].astype(str).to_numpy()

    if verbose:
        nvox = int(masker_global.mask_img_.get_fdata().astype(bool).sum())
        print(
            f'[global] X_all={X_all.shape} (mask vox={nvox}) y={y.shape} groups={groups.shape}'
        )
        print(
            '[global] trials per run:\n',
            pd.Series(groups).value_counts().sort_index().to_string(),
        )

    return df_used, X_all, y, groups, ref_img, masker_global


def make_roi_column_index_map(
    atlas_img: str | Path | nib.Nifti1Image,
    masker_global: NiftiMasker,
    *,
    drop_label: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Map atlas parcel labels to column indices in X_all.

    This uses the fitted `masker_global` to transform the atlas label image into
    the same voxel ordering as X_all's columns.

    Parameters
    ----------
    atlas_img
        3D integer label atlas image (e.g., Schaefer, Harvard–Oxford).
    masker_global
        Fitted NiftiMasker defining the voxel ordering of X_all.
    drop_label
        Label value to ignore (usually 0 for background).
    verbose
        If True, print ROI size percentiles.

    Returns
    -------
    atlas_labels_vec
        Array shape (n_vox_global,) giving the atlas label for each X_all column.
    roi_to_cols
        Dict mapping integer ROI label -> array of column indices into X_all.
    """
    atlas_img = (
        nib.load(str(atlas_img)) if not hasattr(atlas_img, 'get_fdata') else atlas_img
    )

    atlas_vec = masker_global.transform(atlas_img).ravel()
    atlas_labels_vec = np.round(atlas_vec).astype(int)

    labels = np.unique(atlas_labels_vec)
    labels = [int(l) for l in labels if int(l) != drop_label]
    roi_to_cols = {l: np.where(atlas_labels_vec == l)[0] for l in labels}

    if verbose:
        sizes = np.array([len(v) for v in roi_to_cols.values()], dtype=int)
        print(f'[roi map] n_rois={len(roi_to_cols)}')
        if len(sizes):
            print(
                '[roi map] voxels per ROI percentiles:',
                np.percentile(sizes, [0, 5, 50, 95, 100]),
            )

    return atlas_labels_vec, roi_to_cols


def prepare_subject_for_atlas_mvpa(
    behav_bold_df: pd.DataFrame,
    *,
    atlas_img: str | Path | nib.Nifti1Image,
    cfg: Any | None = None,
    group_mask_file: str | Path | None = None,
    y_col: str = 'amount',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    standardize_X: bool = False,
    verbose: bool = False,
) -> SubjectPreparedMVPA:
    """
    Prepare a subject for atlas-based MVPA decoding.

    This function:
    1) extracts global X_all, y, and groups using a group mask
    2) projects the atlas label image into the same voxel ordering as X_all
    3) returns a SubjectPreparedMVPA bundle that downstream decoders can use

    Parameters
    ----------
    behav_bold_df
        Trial table with beta paths and behavioral columns.
    atlas_img
        3D integer label atlas.
    cfg
        Config object; used only if group_mask_file is not provided.
    group_mask_file
        Optional explicit path to a group mask NIfTI.
    y_col, beta_col, group_col
        Column names in behav_bold_df.
    standardize_X
        Whether to have NiftiMasker standardize features (typically False).
    verbose
        Print debug info.

    Returns
    -------
    SubjectPreparedMVPA
        Prepared subject bundle for ROI-wise decoding.
    """
    if group_mask_file is None:
        if cfg is None:
            raise ValueError('Provide cfg or group_mask_file.')
        group_mask_file = (
            cfg.masks_dir
            / 'assess_subject_bold_dropout'
            / 'group_mask_intersection_30pct.nii.gz'
        )

    df_used, X_all, y, groups, ref_img, masker_global = (
        extract_subject_global_Xy_groups(
            behav_bold_df,
            group_mask_file=group_mask_file,
            y_col=y_col,
            beta_col=beta_col,
            group_col=group_col,
            standardize_X=standardize_X,
            verbose=verbose,
        )
    )

    atlas_labels_vec, roi_to_cols = make_roi_column_index_map(
        atlas_img, masker_global, verbose=verbose
    )

    return SubjectPreparedMVPA(
        df_used=df_used,
        X_all=X_all,
        y=y,
        groups=groups,
        ref_img=ref_img,
        masker_global=masker_global,
        atlas_labels_vec=atlas_labels_vec,
        roi_to_cols=roi_to_cols,
    )
