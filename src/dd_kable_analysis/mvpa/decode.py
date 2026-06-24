from __future__ import annotations

"""
High-level decoding routines.

This module contains:
- subject-level atlas ROI decoding (nested group CV ridge regression)
- subject-level atlas ROI decoding for binary choice classification
- helper to paint ROI-level scores back into an atlas image for visualization
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dd_kable_analysis.mvpa.data import build_subject_behav_bold_df
from dd_kable_analysis.mvpa.data_binned import SubjectBinnedData
from dd_kable_analysis.mvpa.features import (
    filter_voxels_runaware,
    prepare_subject_for_atlas_mvpa,
)
from dd_kable_analysis.mvpa.models import (
    nested_groupcv_logreg_predict,
    nested_groupcv_ridge_predict,
    nested_loso_multiclass_logreg_predict,
)


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
    beta_series_subdir: str = 'beta_series',
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
    out = build_subject_behav_bold_df(
        cfg,
        sub_id=sub_id,
        y_col=y_col,
        beta_series_subdir=beta_series_subdir,
        verbose=verbose,
    )
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


def decode_subject_atlas_rois_clf(
    cfg: Any,
    sub_id: str,
    *,
    atlas_img: str | Path | Any,
    y_col: str = 'choseAccept',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    min_voxels_per_roi: int = 50,
    small_thr: float = 1e-4,
    max_small_frac: float = 0.05,
    require_all_runs: bool = True,
    Cs: np.ndarray | None = None,
    verbose: bool = True,
    return_trialwise: bool = False,
    trialwise_rois: set[int] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decode a binary behavioral variable from beta-series patterns within atlas ROIs.

    For each ROI label in `atlas_img`, this function:
      1) extracts trial × voxel data from that ROI
      2) applies run-aware voxel QC
      3) runs nested group CV logistic classification (leave-one-run-out outer CV)
      4) stores out-of-sample classification metrics and optional trialwise outputs

    Parameters
    ----------
    cfg
        Analysis config object (used by build_subject_behav_bold_df and to locate masks).
    sub_id
        Subject ID string.
    atlas_img
        3D integer label atlas (path or Nifti1Image).
    y_col
        Column name in the behavioral/design table to decode. Must be binary 0/1.
    beta_col
        Column name containing beta NIfTI file paths (usually "beta_file").
    group_col
        Column name defining CV grouping (usually "run").
    min_voxels_per_roi
        Skip ROIs with fewer than this many voxels (pre- or post-QC).
    small_thr, max_small_frac, require_all_runs
        Parameters for run-aware voxel QC (`filter_voxels_runaware`).
    Cs
        Logistic regression C grid for nested CV. If None, uses model default.
    verbose
        If True, prints trial/run QA information during subject table creation and prep.
    return_trialwise
        If True, also return a trialwise DataFrame with out-of-sample probabilities/predictions.
    trialwise_rois
        If return_trialwise=True, restrict trialwise output to these ROI labels.
        If None and return_trialwise=True, a ValueError is raised to prevent huge outputs.

    Returns
    -------
    roi_summary_df
        DataFrame with one row per ROI that passes voxel thresholds. Columns include:
        sub_id, roi_label, n_trials, n_runs, n_vox_preQC, n_vox_postQC,
        sensitivity, specificity, balanced_accuracy, roc_auc, log_loss, mean_C.
    (roi_summary_df, trialwise_df)
        If return_trialwise=True, also returns trialwise_df with columns like:
        sub_id, roi_label, run, [trial_type/Delay/amount/choseAccept if present], y,
        p_hat, y_pred.
    """
    if return_trialwise and trialwise_rois is None:
        raise ValueError(
            'return_trialwise=True with trialwise_rois=None will generate a huge '
            'trialwise table (trials × all ROIs). Pass trialwise_rois (set of ints).'
        )

    out = build_subject_behav_bold_df(
        cfg,
        sub_id=sub_id,
        y_col=y_col,
        verbose=verbose,
    )
    behav_bold_df = out.behav_bold_df

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

        y_prob, y_pred, info = nested_groupcv_logreg_predict(
            X_roi_f,
            prep.y,
            prep.groups,
            Cs=Cs,
            verbose=False,
        )

        if return_trialwise and (
            trialwise_rois is None or int(roi_label) in trialwise_rois
        ):
            df_tw = prep.df_used.copy().reset_index(drop=True)

            if 'run' in df_tw.columns:
                df_tw['run'] = df_tw['run'].astype(str)

            df_tw['sub_id'] = str(sub_id)
            df_tw['roi_label'] = int(roi_label)
            df_tw['y'] = prep.y.astype(int)
            df_tw['p_hat'] = y_prob
            df_tw['y_pred'] = y_pred

            extra_cols = [
                c
                for c in ['trial_type', 'Delay', 'amount', 'choseAccept']
                if c in df_tw.columns
            ]
            cols_tw = (
                ['sub_id', 'roi_label', 'run'] + extra_cols + ['y', 'p_hat', 'y_pred']
            )
            cols_tw = [c for c in cols_tw if c in df_tw.columns]
            trialwise_parts.append(df_tw[cols_tw])

        rows.append(
            dict(
                sub_id=str(sub_id),
                roi_label=int(roi_label),
                n_trials=int(len(prep.y)),
                n_runs=int(len(np.unique(prep.groups))),
                n_vox_preQC=n_vox_pre,
                n_vox_postQC=n_vox_post,
                sensitivity=float(info['sensitivity']),
                specificity=float(info['specificity']),
                balanced_accuracy=float(info['balanced_accuracy']),
                accuracy=float(info['accuracy']),
                roc_auc=float(info['roc_auc'])
                if np.isfinite(info['roc_auc'])
                else np.nan,
                log_loss=float(info['log_loss']),
                mean_C=float(np.mean(info['chosen_Cs']))
                if len(info['chosen_Cs'])
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


def decode_value_bins_by_delay(
    subject_data: list[SubjectBinnedData],
    delay_bin: int,
    *,
    n_value_bins: int = 3,
    min_subjects_per_roi: int = 10,
    min_voxels_per_subject: int = 20,
    min_voxels_after_intersection: int = 20,
    Cs: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cross-subject LOSO multiclass decoding for one delay level.

    For each ROI, stacks all subjects' value-bin-averaged patterns (from
    build_subject_binned_roi_patterns) and runs leave-one-subject-out logistic
    regression to classify which value tertile bin (A1/A2/A3) a pattern belongs to.

    The cross-subject voxel intersection is applied per ROI: only voxels that
    passed run-aware QC in every subject are used. This ensures consistent feature
    ordering across subjects.

    Parameters
    ----------
    subject_data
        List of SubjectBinnedData, one per subject, from build_subject_binned_roi_patterns.
    delay_bin
        Delay tertile level to analyse (1, 2, or 3).
    n_value_bins
        Number of value bin classes (default 3 → A1, A2, A3).
    min_subjects_per_roi
        Skip ROIs with fewer valid subjects remaining after per-subject voxel filtering.
    min_voxels_per_subject
        Drop individual subjects from an ROI if their own run-aware QC left fewer than
        this many valid voxels. Prevents a single subject with heavy dropout from
        zeroing out the cross-subject voxel intersection for everyone else.
    min_voxels_after_intersection
        Skip ROIs where the cross-subject voxel intersection (after per-subject filtering)
        has too few voxels.
    Cs
        C grid for inner logistic regression CV. If None, uses the model default.
    verbose
        Print per-ROI progress.

    Returns
    -------
    roi_summary_df
        One row per ROI. Columns:
          roi_label, delay_bin, n_subjects, n_patterns, n_vox,
          chance_level, accuracy, balanced_accuracy, mean_C,
          precision_A1/recall_A1/f1_A1/auc_A1 … (per class).
    subject_preds_df
        One row per subject × value_bin. Columns:
          sub_id, roi_label, delay_bin, value_bin, y_true, y_pred,
          prob_A1/prob_A2/prob_A3.
    confusion_df
        Long-format confusion matrix. Columns:
          roi_label, delay_bin, true_class, pred_class, count.
    """
    # Filter to subjects that have patterns for this delay bin
    valid_subs = [sd for sd in subject_data if delay_bin in sd.roi_patterns]
    n_valid = len(valid_subs)
    if verbose:
        print(f'[delay_bin={delay_bin}] {n_valid}/{len(subject_data)} subjects have data')

    if n_valid < min_subjects_per_roi:
        if verbose:
            print(f'  Too few subjects ({n_valid}); returning empty DataFrames.')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Collect all ROI labels that appear in any valid subject for this delay bin
    all_roi_labels: set[int] = set()
    for sd in valid_subs:
        all_roi_labels.update(sd.roi_patterns[delay_bin].keys())

    roi_summary_rows: list[dict[str, Any]] = []
    subj_pred_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for roi_label in sorted(all_roi_labels):
        # Subjects with patterns AND valid-voxel mask for this ROI and delay bin
        sub_pool_all = [
            sd for sd in valid_subs
            if roi_label in sd.roi_patterns.get(delay_bin, {})
            and roi_label in sd.roi_valid_voxels
        ]

        # Drop subjects whose own QC left too few valid voxels for this ROI.
        # This prevents a single dropout-heavy subject from zeroing the intersection.
        sub_pool = [
            sd for sd in sub_pool_all
            if int(sd.roi_valid_voxels[roi_label].sum()) >= min_voxels_per_subject
        ]
        n_dropped_vox = len(sub_pool_all) - len(sub_pool)
        if verbose and n_dropped_vox > 0:
            print(
                f'  ROI {roi_label}: dropped {n_dropped_vox} subject(s) with '
                f'< {min_voxels_per_subject} valid voxels'
            )

        if len(sub_pool) < min_subjects_per_roi:
            if verbose:
                print(
                    f'  ROI {roi_label}: only {len(sub_pool)} subjects after voxel filter, '
                    f'need {min_subjects_per_roi}, skipping'
                )
            continue

        # Verify consistent total voxel count (same atlas + group mask → should match)
        n_vox_total_vals = set(
            sd.n_vox_total[roi_label] for sd in sub_pool if roi_label in sd.n_vox_total
        )
        if len(n_vox_total_vals) > 1:
            if verbose:
                print(
                    f'  ROI {roi_label}: inconsistent total voxel counts across subjects '
                    f'{n_vox_total_vals}, skipping'
                )
            continue
        n_vox_total = next(iter(n_vox_total_vals))

        # Cross-subject voxel intersection: keep voxels valid in ALL subjects
        common_mask = np.ones(n_vox_total, dtype=bool)
        for sd in sub_pool:
            common_mask &= sd.roi_valid_voxels[roi_label]

        n_vox_common = int(common_mask.sum())
        if n_vox_common < min_voxels_after_intersection:
            if verbose:
                print(
                    f'  ROI {roi_label}: only {n_vox_common} voxels after intersection, '
                    f'need {min_voxels_after_intersection}, skipping'
                )
            continue

        # Build stacked X, y, groups for the LOSO CV
        X_parts: list[np.ndarray] = []
        y_parts: list[int] = []
        grp_parts: list[str] = []

        for sd in sub_pool:
            patterns = sd.roi_patterns[delay_bin][roi_label]  # (n_value_bins, n_vox_total)
            X_parts.append(patterns[:, common_mask].astype(float))  # (n_value_bins, n_vox_common)
            y_parts.extend(range(1, n_value_bins + 1))
            grp_parts.extend([str(sd.sub_id)] * n_value_bins)

        X = np.vstack(X_parts)   # (n_subs * n_value_bins, n_vox_common)
        y = np.array(y_parts, dtype=int)
        groups = np.array(grp_parts)

        if verbose:
            print(
                f'  ROI {roi_label}: n_subs={len(sub_pool)} '
                f'X={X.shape} n_vox_common={n_vox_common}'
            )

        try:
            y_proba, y_pred, info = nested_loso_multiclass_logreg_predict(
                X, y, groups, Cs=Cs, verbose=False
            )
        except Exception as exc:
            if verbose:
                print(f'  ROI {roi_label}: model failed — {exc}')
            continue

        classes = info['classes']

        # --- roi_summary row ---
        row: dict[str, Any] = dict(
            roi_label=int(roi_label),
            delay_bin=int(delay_bin),
            n_subjects=int(len(sub_pool)),
            n_patterns=int(len(y)),
            n_vox=int(n_vox_common),
            chance_level=float(info['chance_level']),
            accuracy=float(info['accuracy']),
            balanced_accuracy=float(info['balanced_accuracy']),
            mean_C=float(info['mean_C']),
        )
        for j, cls in enumerate(classes):
            row[f'precision_A{cls}'] = float(info['precision'][j])
            row[f'recall_A{cls}'] = float(info['recall'][j])
            row[f'f1_A{cls}'] = float(info['f1'][j])
            auc_val = info['auc_per_class'][j]
            row[f'auc_A{cls}'] = float(auc_val) if np.isfinite(auc_val) else np.nan
        roi_summary_rows.append(row)

        # --- per-subject prediction rows ---
        for i, sd in enumerate(sub_pool):
            for v_idx in range(n_value_bins):
                flat_i = i * n_value_bins + v_idx
                pred_row: dict[str, Any] = dict(
                    sub_id=str(sd.sub_id),
                    roi_label=int(roi_label),
                    delay_bin=int(delay_bin),
                    value_bin=int(y[flat_i]),
                    y_true=int(y[flat_i]),
                    y_pred=int(y_pred[flat_i]),
                )
                for j, cls in enumerate(classes):
                    pred_row[f'prob_A{cls}'] = float(y_proba[flat_i, j])
                subj_pred_rows.append(pred_row)

        # --- confusion matrix (long format) ---
        cm = np.array(info['confusion_matrix'])
        for ti, true_cls in enumerate(classes):
            for pi, pred_cls in enumerate(classes):
                confusion_rows.append(
                    dict(
                        roi_label=int(roi_label),
                        delay_bin=int(delay_bin),
                        true_class=int(true_cls),
                        pred_class=int(pred_cls),
                        count=int(cm[ti, pi]),
                    )
                )

    roi_summary_df = (
        pd.DataFrame(roi_summary_rows).sort_values('roi_label').reset_index(drop=True)
        if roi_summary_rows
        else pd.DataFrame()
    )
    subject_preds_df = pd.DataFrame(subj_pred_rows) if subj_pred_rows else pd.DataFrame()
    confusion_df = pd.DataFrame(confusion_rows) if confusion_rows else pd.DataFrame()

    return roi_summary_df, subject_preds_df, confusion_df


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
