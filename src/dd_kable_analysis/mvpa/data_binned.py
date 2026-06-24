from __future__ import annotations

"""
Data preparation for value-bin-by-delay MVPA analysis.

Builds per-subject delay × value tertile bin-averaged beta patterns for
cross-subject LOSO multiclass classification.

Typical usage
-------------
1. Call build_subject_behav_bold_df for all subjects to get behavioral tables.
2. Call compute_global_bin_edges on the pooled behavioral tables.
3. Call build_subject_binned_roi_patterns for each subject (passing the
   pre-built behav_bold_result to avoid double-loading).
4. Pass the list of SubjectBinnedData to decode_value_bins_by_delay in decode.py.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dd_kable_analysis.mvpa.data import SubjectBehavBoldResult, build_subject_behav_bold_df
from dd_kable_analysis.mvpa.features import (
    filter_voxels_runaware,
    prepare_subject_for_atlas_mvpa,
)


def compute_global_bin_edges(
    trial_dfs: list[pd.DataFrame],
    *,
    delay_col: str = 'Delay',
    value_col: str = 'amount',
    n_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pooled-across-subjects quantile edges for delay and value.

    All subjects' trials are pooled before computing percentile cut-points so
    that bin boundaries are consistent across subjects.

    Parameters
    ----------
    trial_dfs
        List of per-subject trial DataFrames (must contain delay_col and value_col).
    delay_col, value_col
        Column names to bin.
    n_bins
        Number of equal-frequency bins (default 3 = tertiles).

    Returns
    -------
    delay_edges, value_edges
        Interior quantile edges of shape (n_bins - 1,).
        Pass both to assign_bin_labels or np.digitize(x, edges) + 1 to get
        1-indexed bin labels.
    """
    delays = pd.concat([df[delay_col] for df in trial_dfs]).dropna().to_numpy(float)
    values = pd.concat([df[value_col] for df in trial_dfs]).dropna().to_numpy(float)

    q = np.linspace(0, 100, n_bins + 1)[1:-1]  # interior percentiles, e.g. [33.33, 66.67]
    delay_edges = np.percentile(delays, q)
    value_edges = np.percentile(values, q)
    return delay_edges, value_edges


def assign_bin_labels(
    df: pd.DataFrame,
    delay_edges: np.ndarray,
    value_edges: np.ndarray,
    *,
    delay_col: str = 'Delay',
    value_col: str = 'amount',
) -> pd.DataFrame:
    """
    Add 1-indexed delay_bin and value_bin columns to a trial DataFrame.

    Uses np.digitize against the interior edges from compute_global_bin_edges.
    A trial with x < delay_edges[0] → delay_bin=1,
    delay_edges[0] <= x < delay_edges[1] → delay_bin=2, etc.

    Parameters
    ----------
    df
        Trial DataFrame with delay_col and value_col present.
    delay_edges, value_edges
        Interior bin edges (n_bins - 1,) from compute_global_bin_edges.

    Returns
    -------
    Copy of df with new integer columns 'delay_bin' and 'value_bin'.
    """
    df = df.copy()
    df['delay_bin'] = np.digitize(df[delay_col].to_numpy(float), delay_edges) + 1
    df['value_bin'] = np.digitize(df[value_col].to_numpy(float), value_edges) + 1
    return df


@dataclass
class SubjectBinnedData:
    """
    Per-subject binned pattern data for value-bin-by-delay MVPA.

    Attributes
    ----------
    sub_id
        Subject identifier.
    roi_patterns
        delay_bin → roi_label → float32 array of shape (n_value_bins, n_vox_roi).
        Each row is the mean beta pattern across all trials in that value bin at
        that delay level. Uses ALL ROI voxels (no per-subject QC filtering here);
        cross-subject voxel intersection is applied at decode time via roi_valid_voxels.
    roi_valid_voxels
        roi_label → bool mask of shape (n_vox_roi,).
        True where a voxel passed run-aware QC for this subject. At decode time,
        the AND across all subjects gives the common valid voxel set per ROI.
    bin_trial_counts
        delay_bin → roi_label → int array of shape (n_value_bins,).
        Number of trials contributing to each bin average.
    n_vox_total
        roi_label → total voxels in ROI within the group mask (same for all subjects).
    excluded_delay_bins
        Delay bins dropped for this subject because at least one value-bin cell
        had fewer than min_trials_per_bin trials.
    qc
        Basic trial-count QC from build_subject_behav_bold_df.
    """

    sub_id: str
    roi_patterns: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    roi_valid_voxels: dict[int, np.ndarray] = field(default_factory=dict)
    bin_trial_counts: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    n_vox_total: dict[int, int] = field(default_factory=dict)
    excluded_delay_bins: list[int] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)


def build_subject_binned_roi_patterns(
    cfg: Any,
    sub_id: str,
    atlas_img: Any,
    delay_edges: np.ndarray,
    value_edges: np.ndarray,
    *,
    behav_bold_result: SubjectBehavBoldResult | None = None,
    n_value_bins: int = 3,
    min_trials_per_bin: int = 3,
    min_voxels_per_roi: int = 50,
    small_thr: float = 1e-4,
    max_small_frac: float = 0.05,
    require_all_runs: bool = True,
    beta_series_subdir: str = 'beta_series',
    verbose: bool = False,
) -> SubjectBinnedData:
    """
    Build per-ROI per-delay-bin averaged beta patterns for one subject.

    For each valid delay bin, trials are split by value bin and averaged to produce
    one brain pattern per value bin. Patterns are stored unfiltered (all ROI voxels)
    alongside a per-subject valid-voxel mask so that the cross-subject intersection
    can be computed at decode time.

    Parameters
    ----------
    cfg
        Analysis config object with paths.
    sub_id
        Subject identifier.
    atlas_img
        Integer-label atlas image (path or Nifti1Image).
    delay_edges, value_edges
        Interior bin edges from compute_global_bin_edges.
    behav_bold_result
        Optional pre-built SubjectBehavBoldResult. When provided, the behavioral
        data loading step is skipped (avoids a redundant filesystem pass when the
        entry-point script already loaded it for bin-edge computation).
    n_value_bins
        Number of value bins (must equal len(value_edges) + 1).
    min_trials_per_bin
        Minimum trials required in every value-bin cell within a delay bin.
        Delay bins failing this for any cell are excluded for this subject.
    min_voxels_per_roi
        ROIs with fewer total voxels in the group mask are skipped entirely.
    small_thr, max_small_frac, require_all_runs
        Run-aware voxel QC parameters (same semantics as the ridge analysis).
    beta_series_subdir
        Subdirectory under output_root for beta series images.
    verbose
        Print trial/run QA info.

    Returns
    -------
    SubjectBinnedData
    """
    if behav_bold_result is None:
        out = build_subject_behav_bold_df(
            cfg,
            sub_id=sub_id,
            y_col='amount',
            beta_series_subdir=beta_series_subdir,
            verbose=verbose,
            strict=False,
        )
    else:
        out = behav_bold_result

    behav_bold_df = out.behav_bold_df

    prep = prepare_subject_for_atlas_mvpa(
        behav_bold_df,
        atlas_img=atlas_img,
        cfg=cfg,
        y_col='amount',
        verbose=verbose,
    )

    # Assign bin labels to df_used (rows aligned 1:1 with prep.X_all)
    df_binned = assign_bin_labels(
        prep.df_used,
        delay_edges=delay_edges,
        value_edges=value_edges,
    )
    delay_bin_arr = df_binned['delay_bin'].to_numpy(int)
    value_bin_arr = df_binned['value_bin'].to_numpy(int)

    result = SubjectBinnedData(
        sub_id=str(sub_id),
        qc={
            'n_trials_before_vif': int(out.n_trials_before_vif),
            'n_trials_after_vif': int(out.n_trials_after_vif_and_missing),
            'n_missing_betas': int(out.n_missing_betas),
            'n_high_vif_omitted': int(out.n_high_vif_omitted),
        },
    )

    n_delay_bins = int(len(delay_edges)) + 1

    # Determine which delay bins have sufficient trials in all value-bin cells.
    # This is a subject-level check (same regardless of ROI).
    valid_delay_bins: list[int] = []
    excluded_delay_bins: list[int] = []
    for d_bin in range(1, n_delay_bins + 1):
        delay_mask = delay_bin_arr == d_bin
        counts = np.array([
            int(np.sum(delay_mask & (value_bin_arr == v)))
            for v in range(1, n_value_bins + 1)
        ])
        if np.any(counts < min_trials_per_bin):
            excluded_delay_bins.append(d_bin)
            if verbose:
                print(
                    f'[{sub_id}] delay_bin={d_bin}: cell counts {counts.tolist()} '
                    f'< min {min_trials_per_bin}, excluding'
                )
        else:
            valid_delay_bins.append(d_bin)

    result.excluded_delay_bins = excluded_delay_bins

    # Pre-initialise roi_patterns containers for valid delay bins
    for d_bin in valid_delay_bins:
        result.roi_patterns[d_bin] = {}
        result.bin_trial_counts[d_bin] = {}

    # Per-ROI: compute voxel QC mask and bin-averaged patterns
    for roi_label, col_indices in prep.roi_to_cols.items():
        if col_indices.size == 0:
            continue

        X_roi = prep.X_all[:, col_indices]  # (n_trials, n_vox_roi) — unfiltered
        n_vox = int(X_roi.shape[1])
        if n_vox < min_voxels_per_roi:
            continue

        result.n_vox_total[roi_label] = n_vox

        # Compute per-subject valid-voxel mask (but do NOT filter X_roi here)
        try:
            _, qc_info = filter_voxels_runaware(
                X_roi,
                prep.groups,
                small_thr=small_thr,
                max_small_frac=max_small_frac,
                require_all_runs=require_all_runs,
                verbose=False,
            )
            result.roi_valid_voxels[roi_label] = qc_info.keep_mask
        except RuntimeError:
            # All voxels failed QC for this subject/ROI — skip the ROI entirely.
            continue

        # Compute bin-averaged patterns for each valid delay bin
        for d_bin in valid_delay_bins:
            delay_mask = delay_bin_arr == d_bin
            patterns = np.zeros((n_value_bins, n_vox), dtype=np.float32)
            counts = np.zeros(n_value_bins, dtype=int)
            for v_idx, v_bin in enumerate(range(1, n_value_bins + 1)):
                bin_mask = delay_mask & (value_bin_arr == v_bin)
                patterns[v_idx] = X_roi[bin_mask].mean(axis=0)
                counts[v_idx] = int(bin_mask.sum())
            result.roi_patterns[d_bin][roi_label] = patterns
            result.bin_trial_counts[d_bin][roi_label] = counts

    return result
