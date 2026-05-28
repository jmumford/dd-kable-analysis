from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dd_kable_analysis.mvpa import build_subject_behav_bold_df
from dd_kable_analysis.mvpa.features import (
    filter_voxels_runaware,
    prepare_subject_for_atlas_mvpa,
)

# =============================
# Core metric
# =============================


def cross_run_bin_pattern_consistency(
    X_roi: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    *,
    n_bins: int = 4,
    min_trials_per_bin: int = 3,
    return_patterns: bool = False,
) -> dict[str, Any]:
    """
    Within each run:
      - bin y into quantiles (pd.qcut)
      - average voxel patterns per bin

    Across runs:
      - correlate matching bin-mean patterns across all run pairs (corr across voxels)
      - average across bins and run pairs

    Returns dict with:
      mean_r, pair_rs, pair_names, kept_runs, n_bin_corrs_used, n_bins_used_by_run
      optionally patterns (run -> (nb, n_vox)).
    """
    X_roi = np.asarray(X_roi)
    y = np.asarray(y, float)
    runs = np.asarray(runs).astype(str)

    uniq_runs = np.unique(runs)
    patterns: dict[str, np.ndarray] = {}
    n_bins_used_by_run: dict[str, int] = {}

    for r in uniq_runs:
        m = runs == r
        yr = y[m]
        Xr = X_roi[m, :]

        # quantile bins within run (pseudo-repeats)
        bins = pd.qcut(yr, q=n_bins, labels=False, duplicates='drop')
        nb = int(np.max(bins)) + 1 if len(bins) else 0
        if nb < 2:
            continue

        pats = []
        for b in range(nb):
            mb = np.asarray(bins) == b
            if int(mb.sum()) < min_trials_per_bin:
                pats = None
                break
            pats.append(Xr[mb].mean(axis=0))

        if pats is not None:
            patterns[str(r)] = np.vstack(pats)  # (nb, n_vox)
            n_bins_used_by_run[str(r)] = int(nb)

    kept = sorted(patterns.keys())
    if len(kept) < 2:
        out: dict[str, Any] = dict(
            mean_r=np.nan,
            pair_rs=[],
            pair_names=[],
            kept_runs=kept,
            n_bin_corrs_used=0,
            n_bins_used_by_run=n_bins_used_by_run,
        )
        if return_patterns:
            out['patterns'] = patterns
        return out

    pair_rs: list[float] = []
    pair_names: list[tuple[str, str]] = []
    n_bin_corrs_used = 0

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            r1, r2 = kept[i], kept[j]
            P1, P2 = patterns[r1], patterns[r2]
            nb = min(P1.shape[0], P2.shape[0])

            rs = []
            for b in range(nb):
                v1, v2 = P1[b], P2[b]
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                rs.append(float(np.corrcoef(v1, v2)[0, 1]))

            if rs:
                pair_rs.append(float(np.mean(rs)))
                pair_names.append((r1, r2))
                n_bin_corrs_used += len(rs)

    out = dict(
        mean_r=float(np.mean(pair_rs)) if pair_rs else np.nan,
        pair_rs=pair_rs,
        pair_names=pair_names,
        kept_runs=kept,
        n_bin_corrs_used=int(n_bin_corrs_used),
        n_bins_used_by_run=n_bins_used_by_run,
    )
    if return_patterns:
        out['patterns'] = patterns
    return out


def cross_run_bin_pattern_consistency_with_null(
    X_roi: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    *,
    n_bins: int = 4,
    min_trials_per_bin: int = 3,
    n_perm: int = 200,
    seed: int = 0,
    return_null: bool = False,
) -> dict[str, Any]:
    """
    Observed stability + within-run shuffle null.

    Null: permute y within each run (preserves run structure and trial counts but
    breaks any relationship between y and patterns).
    """
    rng = np.random.default_rng(seed)
    runs = np.asarray(runs).astype(str)
    y = np.asarray(y, float)

    obs = cross_run_bin_pattern_consistency(
        X_roi,
        y,
        runs,
        n_bins=n_bins,
        min_trials_per_bin=min_trials_per_bin,
        return_patterns=False,
    )
    obs_r = float(obs['mean_r'])

    null_rs = np.full(int(n_perm), np.nan, dtype=float)
    uniq_runs = np.unique(runs)

    for p in range(int(n_perm)):
        y_perm = y.copy()
        for r in uniq_runs:
            m = runs == r
            y_perm[m] = rng.permutation(y_perm[m])

        null = cross_run_bin_pattern_consistency(
            X_roi,
            y_perm,
            runs,
            n_bins=n_bins,
            min_trials_per_bin=min_trials_per_bin,
            return_patterns=False,
        )
        null_rs[p] = float(null['mean_r'])

    finite = np.isfinite(null_rs)
    null_mean = float(np.mean(null_rs[finite])) if np.any(finite) else np.nan
    null_sd = float(np.std(null_rs[finite])) if np.any(finite) else np.nan

    # empirical one-sided p-value: P(null >= obs)
    p_emp = (
        float((np.sum(null_rs[finite] >= obs_r) + 1) / (np.sum(finite) + 1))
        if np.any(finite) and np.isfinite(obs_r)
        else np.nan
    )

    out: dict[str, Any] = dict(
        mean_r_obs=obs_r,
        mean_r_null_mean=null_mean,
        mean_r_null_sd=null_sd,
        p_emp=p_emp,
        n_perm=int(n_perm),
    )
    if return_null:
        out['null_rs'] = null_rs.tolist()
    return out


# =============================
# Subject-level wrapper (atlas ROI label)
# =============================


def noise_ceiling_subject_roi(
    cfg: Any,
    sub_id: str,
    *,
    atlas_img: str | Path | Any,  # nifti path or Nifti1Image
    roi_label: int,  # integer label within atlas_img
    out_root: str | Path | None = None,
    out_tag: str = 'noise_ceiling',
    y_col: str = 'amount',
    beta_col: str = 'beta_file',
    group_col: str = 'run',
    # voxel QC:
    apply_voxel_qc: bool = True,
    small_thr: float = 1e-4,
    max_small_frac: float = 0.05,
    require_all_runs: bool = True,
    min_voxels: int = 1,
    # binning:
    n_bins: int = 4,
    min_trials_per_bin: int = 3,
    return_patterns: bool = False,
    # shuffle-null:
    compute_null: bool = False,
    n_perm: int = 200,
    seed: int = 0,
    return_null: bool = False,
    # misc:
    verbose: bool = False,
    save_json: bool = True,
) -> dict[str, Any]:
    """
    Compute cross-run binned-pattern consistency for one subject and one ROI label.

    If compute_null=True, returns observed mean_r plus null summary stats:
      mean_r_obs, mean_r_null_mean, mean_r_null_sd, p_emp, n_perm
    else returns:
      mean_r, pair_rs, pair_names, kept_runs, n_bin_corrs_used, ...

    Optionally writes:
      {out_root}/{out_tag}/sub-{sub_id}/roi-{roi_label}_ceiling.json
    """
    out = build_subject_behav_bold_df(cfg, sub_id=sub_id, strict=True, verbose=verbose)

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

    cols = prep.roi_to_cols.get(int(roi_label), None)
    base = dict(
        sub_id=str(sub_id),
        roi_label=int(roi_label),
        y_col=str(y_col),
        n_trials=int(len(prep.y)),
        n_runs=int(len(np.unique(prep.groups.astype(str)))),
    )

    if cols is None or len(cols) == 0:
        res = dict(
            **base,
            n_vox=0,
            mean_r=np.nan,
            n_bin_corrs_used=0,
            error='roi_label not found in atlas after masking',
        )
        _maybe_save(res, cfg, out_root, out_tag, save_json=save_json)
        return res

    X_roi = prep.X_all[:, cols]
    n_vox_pre = int(X_roi.shape[1])

    if n_vox_pre < min_voxels:
        res = dict(
            **base,
            n_vox=n_vox_pre,
            n_vox_preQC=n_vox_pre,
            n_vox_postQC=n_vox_pre,
            mean_r=np.nan,
            n_bin_corrs_used=0,
            error=f'ROI has too few voxels (min_voxels={min_voxels})',
        )
        _maybe_save(res, cfg, out_root, out_tag, save_json=save_json)
        return res

    if apply_voxel_qc:
        try:
            X_roi, _qc = filter_voxels_runaware(
                X_roi,
                prep.groups,
                small_thr=small_thr,
                max_small_frac=max_small_frac,
                require_all_runs=require_all_runs,
                verbose=False,
            )
            n_vox_post = int(X_roi.shape[1])
        except RuntimeError as e:
            res = dict(
                **base,
                n_vox=n_vox_pre,
                n_vox_preQC=n_vox_pre,
                n_vox_postQC=0,
                mean_r=np.nan,
                n_bin_corrs_used=0,
                error=f'voxel_qc_failed: {e}',
            )
            _maybe_save(res, cfg, out_root, out_tag, save_json=save_json)
            return res
    else:
        n_vox_post = n_vox_pre

    if n_vox_post < min_voxels:
        res = dict(
            **base,
            n_vox=n_vox_post,
            n_vox_preQC=n_vox_pre,
            n_vox_postQC=n_vox_post,
            mean_r=np.nan,
            n_bin_corrs_used=0,
            error=f'too_few_voxels_after_qc (min_voxels={min_voxels})',
        )
        _maybe_save(res, cfg, out_root, out_tag, save_json=save_json)
        return res

    runs = prep.groups.astype(str)

    if compute_null:
        metric = cross_run_bin_pattern_consistency_with_null(
            X_roi,
            prep.y,
            runs,
            n_bins=n_bins,
            min_trials_per_bin=min_trials_per_bin,
            n_perm=n_perm,
            seed=seed,
            return_null=return_null,
        )
        metric_out = metric  # already excludes patterns/pairs
    else:
        metric = cross_run_bin_pattern_consistency(
            X_roi,
            prep.y,
            runs,
            n_bins=n_bins,
            min_trials_per_bin=min_trials_per_bin,
            return_patterns=return_patterns,
        )
        metric_out = metric

    res = dict(
        **base,
        n_vox=int(X_roi.shape[1]),
        n_vox_preQC=int(n_vox_pre),
        n_vox_postQC=int(n_vox_post),
        n_bins=int(n_bins),
        min_trials_per_bin=int(min_trials_per_bin),
        apply_voxel_qc=bool(apply_voxel_qc),
        voxel_qc_params=(
            dict(
                small_thr=float(small_thr),
                max_small_frac=float(max_small_frac),
                require_all_runs=bool(require_all_runs),
            )
            if apply_voxel_qc
            else None
        ),
        **{k: v for k, v in metric_out.items() if k != 'patterns'},
    )

    if (not compute_null) and return_patterns:
        res['patterns'] = metric.get('patterns', {})

    _maybe_save(res, cfg, out_root, out_tag, save_json=save_json)
    return res


def _maybe_save(
    res: dict[str, Any],
    cfg: Any,
    out_root: str | Path | None,
    out_tag: str,
    *,
    save_json: bool = True,
) -> None:
    if not save_json:
        return
    import json

    root = Path(out_root) if out_root is not None else Path(cfg.output_root)
    out_dir = root / out_tag / f'sub-{res["sub_id"]}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'roi-{int(res["roi_label"]):04d}_ceiling.json'

    def _to_jsonable(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return list(x)
        return x

    clean = {k: _to_jsonable(v) for k, v in res.items()}
    with open(out_file, 'w') as f:
        json.dump(clean, f, indent=2, sort_keys=True)
