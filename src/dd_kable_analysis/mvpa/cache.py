from __future__ import annotations

"""
Caching utilities for MVPA.

Purpose:
- Avoid re-running NiftiMasker.transform (expensive NIfTI I/O) when doing many
  permutation refits or repeated decoding runs.
- Cache per-subject global X_all, y, groups, and atlas label vector.

The cache is saved as a compressed NPZ file containing numpy arrays plus a
minimal copy of df_used stored as column arrays.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dd_kable_analysis.mvpa.data import build_subject_behav_bold_df
from dd_kable_analysis.mvpa.features import prepare_subject_for_atlas_mvpa


@dataclass
class SubjectMVPAFeatureCache:
    """Loaded per-subject cached features for fast decoding/permutation tests."""

    sub_id: str
    X_all: np.ndarray  # float32 (n_trials, n_vox_global)
    y: np.ndarray  # float32 (n_trials,)
    groups: np.ndarray  # run labels as strings (n_trials,)
    atlas_labels_vec: np.ndarray  # int32 (n_vox_global,)
    df_used: pd.DataFrame  # minimal trial metadata corresponding to X_all rows


def get_subject_cache_path(
    cfg: Any,
    sub_id: str,
    *,
    cache_root: str = 'mvpa_cache',
    cache_tag: str = 'schaefer200_2mm_amount',
    filename: str = 'prep_cache.npz',
) -> Path:
    """
    Construct the path to the per-subject cache file.

    Parameters
    ----------
    cfg
        Config object; must have cfg.output_root.
    sub_id
        Subject ID.
    cache_root
        Top-level directory under cfg.output_root for MVPA caches.
    cache_tag
        Analysis identifier (e.g., atlas + resolution + target), used to separate caches.
    filename
        Cache filename.

    Returns
    -------
    Path
        Full path to the NPZ cache file.
    """
    return Path(cfg.output_root) / cache_root / cache_tag / f'sub-{sub_id}' / filename


def save_subject_prep_cache(
    cfg: Any,
    sub_id: str,
    *,
    atlas_img: Any,
    y_col: str = 'amount',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    cache_root: str = 'mvpa_cache',
    cache_tag: str = 'schaefer200_2mm_amount',
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Build and save a per-subject MVPA feature cache.

    This runs the expensive feature extraction step (NiftiMasker.transform on all
    trial beta images) once and saves the resulting arrays. Useful for permutation
    tests where you want to refit many models with shuffled y.

    Parameters
    ----------
    cfg
        Config object.
    sub_id
        Subject ID.
    atlas_img
        Label atlas image (path or Nifti1Image). Used only to compute atlas_labels_vec
        consistent with X_all voxel ordering.
    y_col, beta_col, group_col
        Column names used by the subject trial table and extraction.
    cache_root, cache_tag
        Where to write the cache.
    overwrite
        If True, overwrite an existing cache file.
    verbose
        Print status.

    Returns
    -------
    Path
        Path to the written NPZ file.
    """
    cache_path = get_subject_cache_path(
        cfg, sub_id, cache_root=cache_root, cache_tag=cache_tag
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not overwrite:
        if verbose:
            print(f'[cache] exists, skipping: {cache_path}')
        return cache_path

    out = build_subject_behav_bold_df(cfg, sub_id=sub_id, verbose=verbose, strict=True)
    prep = prepare_subject_for_atlas_mvpa(
        out.behav_bold_df,
        atlas_img=atlas_img,
        cfg=cfg,
        y_col=y_col,
        beta_col=beta_col,
        group_col=group_col,
        standardize_X=False,
        verbose=verbose,
    )

    # Save arrays (use compact dtypes)
    X_all = np.asarray(prep.X_all, dtype=np.float32)
    y = np.asarray(prep.y, dtype=np.float32)
    groups = np.asarray(prep.groups).astype('U32')  # fixed-width unicode strings
    atlas_labels_vec = np.asarray(prep.atlas_labels_vec, dtype=np.int32)

    # df_used: keep only what you'll want later
    df_used = prep.df_used.copy()
    keep_cols = [
        c
        for c in ['trial_type', 'run', y_col, 'delay', 'choseAccept']
        if c in df_used.columns
    ]
    df_used = df_used[keep_cols].reset_index(drop=True)

    np.savez_compressed(
        cache_path,
        sub_id=str(sub_id),
        X_all=X_all,
        y=y,
        groups=groups,
        atlas_labels_vec=atlas_labels_vec,
        df_cols=np.array(df_used.columns.tolist(), dtype='U32'),
        **{f'df_{c}': np.asarray(df_used[c].to_numpy()) for c in df_used.columns},
    )

    if verbose:
        print(
            f'[cache] wrote: {cache_path} X_all={X_all.shape} labels={atlas_labels_vec.shape}'
        )

    return cache_path


def load_subject_prep_cache(cache_path: str | Path) -> SubjectMVPAFeatureCache:
    """
    Load a previously saved per-subject MVPA feature cache.

    Parameters
    ----------
    cache_path
        Path to NPZ cache file.

    Returns
    -------
    SubjectMVPAFeatureCache
        Loaded cache object.
    """
    z = np.load(str(cache_path), allow_pickle=False)
    sub_id = str(z['sub_id'])
    X_all = z['X_all']
    y = z['y']
    groups = z['groups'].astype(str)
    atlas_labels_vec = z['atlas_labels_vec']

    cols = [str(c) for c in z['df_cols'].tolist()]
    df_dict = {c: z[f'df_{c}'] for c in cols}
    df_used = pd.DataFrame(df_dict)

    return SubjectMVPAFeatureCache(
        sub_id=sub_id,
        X_all=X_all,
        y=y,
        groups=groups,
        atlas_labels_vec=atlas_labels_vec,
        df_used=df_used,
    )


def roi_to_cols_from_atlas_labels(
    atlas_labels_vec: np.ndarray,
    drop_label: int = 0,
) -> dict[int, np.ndarray]:
    """
    Create ROI->column-index mapping from an atlas label vector.

    Parameters
    ----------
    atlas_labels_vec
        Vector of length n_vox_global giving atlas label per voxel/column.
    drop_label
        Background label value to exclude (usually 0).

    Returns
    -------
    dict
        Mapping from ROI integer label to numpy array of column indices.
    """
    labels = np.unique(atlas_labels_vec.astype(int))
    labels = [int(l) for l in labels if int(l) != drop_label]
    return {l: np.where(atlas_labels_vec == l)[0] for l in labels}
