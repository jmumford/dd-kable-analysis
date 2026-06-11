from __future__ import annotations

"""
Data assembly utilities for MVPA decoding.

This module constructs a per-subject trial table that links:
- behavioral/design-matrix trial rows
- beta-series NIfTI files for each trial regressor
- run labels for group-wise CV

It also applies trial omissions (high-VIF trials, missing beta files) and
enforces minimum runs/trials-per-run requirements.
"""

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from dd_kable_analysis.tseries_model.design_matrix import make_design_matrix


def load_initial_and_vif_tables(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load subject×run QA table and high-VIF trial omission table.

    Returns
    -------
    initial_sub_run_df
        DataFrame of QA-passed subject/run combinations.
    high_vif_sub_run_df
        DataFrame listing trials to omit due to high VIF.
    """
    initial_sub_run_file = (
        cfg.subject_lists / 'initial_qa_pass_and_mask_pass_subjects_runs.csv'
    )
    initial_sub_run_df = pd.read_csv(initial_sub_run_file)

    high_vif_sub_run_file = (
        cfg.data_root
        / 'scripts'
        / 'dd-kable-analysis'
        / 'analyses'
        / 'beta_series_analysis'
        / 'subject_lists'
        / 'vif_gt_5.csv'
    )
    high_vif_sub_run_df = pd.read_csv(high_vif_sub_run_file)
    return initial_sub_run_df, high_vif_sub_run_df


def make_high_vif_trial_set(
    high_vif_sub_run_df: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    """
    Convert a high-VIF omission table into a set of keys for fast filtering.

    Parameters
    ----------
    high_vif_sub_run_df
        Must contain columns: sub_id, run, trial
        where `trial` matches behav_data['trial_type'] values like 'trial00'.

    Returns
    -------
    set
        Set of (sub_id, run, trial_type) tuples to omit.
    """
    required = {'sub_id', 'run', 'trial'}
    missing = required - set(high_vif_sub_run_df.columns)
    if missing:
        raise ValueError(f'vif_gt_5.csv missing columns: {missing}')

    return {
        (str(r.sub_id), str(r.run), str(r.trial))
        for r in high_vif_sub_run_df.itertuples(index=False)
    }


def get_subject_good_runs(
    initial_sub_run_df: pd.DataFrame,
    sub_id: str,
    sub_col: str = 'sub_id',
    run_col: str = 'run',
) -> list[str]:
    """
    Return list of run IDs (strings) that passed initial QA for a subject.

    Parameters
    ----------
    initial_sub_run_df
        QA table containing at least sub_col and run_col.
    sub_id
        Subject identifier.
    sub_col, run_col
        Column names in initial_sub_run_df.

    Returns
    -------
    list[str]
        Sorted unique run IDs. Empty if subject not present.
    """
    if sub_col not in initial_sub_run_df or run_col not in initial_sub_run_df:
        raise ValueError(f'initial_sub_run_df must have columns {sub_col}, {run_col}')

    runs = (
        initial_sub_run_df.loc[
            initial_sub_run_df[sub_col].astype(str) == str(sub_id), run_col
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(runs)


@dataclass
class SubjectBehavBoldResult:
    """
    Result of building the subject-level trial table linking behavior and betas.
    """

    behav_bold_df: pd.DataFrame
    good_runs: list[str]
    n_trials_before_vif: int
    n_trials_after_vif_and_missing: int
    n_missing_betas: int
    n_high_vif_omitted: int
    n_missing_sv_rows: int = 0
    trials_kept_by_run: dict[str, int] | None = None
    runs_passing_trial_threshold: list[str] | None = None


def _is_sv_target(y_col: str) -> bool:
    return y_col in {'SV_LL', 'SV_chosen', 'SV_unchosen', 'SV_SS', 'DV'}


def _run_to_token(run: str) -> str:
    try:
        return f'run-{int(float(str(run))):02d}'
    except ValueError:
        return f'run-{run}'


def _find_sv_file(cfg: Any, sub_id: str, run: str, sv_dirname: str) -> Path | None:
    sv_root = Path(cfg.output_root) / sv_dirname
    sub_label = str(sub_id).replace('sub-', '')
    sub_dir = sv_root / f'sub-{sub_label}'

    if not sub_dir.exists():
        return None

    run_token = _run_to_token(run)
    patterns = [
        sub_dir
        / 'ses-*'
        / 'func'
        / f'sub-{sub_label}_ses-*_task-dd_{run_token}_desc-subjectivevalue_events.tsv',
        sub_dir
        / 'func'
        / f'sub-{sub_label}_task-dd_{run_token}_desc-subjectivevalue_events.tsv',
    ]

    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob(str(pat)))

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(
            f'Multiple SV derivative files found for sub-{sub_label}, run={run}: {matches}'
        )

    return Path(matches[0])


def _merge_subjective_value_columns(
    behav_bold_run: pd.DataFrame,
    cfg: Any,
    sub_id: str,
    run: str,
    *,
    y_col: str,
    sv_dirname: str,
    strict: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, int]:
    """
    Merge subjective-value columns onto a single subject/run trial table.

    The merge is keyed by rounded onset, because the SV derivative files are
    generated separately from the BIDS events tables but should align trial by
    trial within each run.

    Returns
    -------
    behav_bold_run
        Input table with the subjective-value columns merged in when available.
    n_missing_sv_rows
        Number of rows missing the requested target column after merge.
    """
    sv_file = _find_sv_file(cfg, sub_id=sub_id, run=str(run), sv_dirname=sv_dirname)

    if sv_file is None:
        msg = (
            f'[{sub_id}] missing subjective-value derivative file for run {run} '
            f'under output_root/{sv_dirname}.'
        )
        if strict:
            raise ValueError(msg)
        if verbose:
            print('WARNING:', msg)
        return behav_bold_run.iloc[0:0].copy(), 0

    sv_df = pd.read_csv(sv_file, sep='\t')
    if 'onset' not in sv_df.columns:
        raise ValueError(f'Subjective-value file missing onset column: {sv_file}')

    behav_bold_run = behav_bold_run.copy()
    sv_df = sv_df.copy()

    behav_bold_run['_onset_key'] = behav_bold_run['onset'].round(6)
    sv_df['_onset_key'] = sv_df['onset'].round(6)

    sv_cols = ['_onset_key', 'SV_LL', 'SV_chosen', 'SV_unchosen', 'SV_SS', 'DV']
    sv_cols = [c for c in sv_cols if c in sv_df.columns]
    sv_df = sv_df[sv_cols].drop_duplicates(subset=['_onset_key'], keep='first')

    merged = behav_bold_run.merge(
        sv_df,
        on='_onset_key',
        how='left',
        validate='many_to_one',
    ).drop(columns=['_onset_key'])

    if y_col not in merged.columns:
        raise ValueError(
            f"SV merge finished but y_col='{y_col}' is absent for sub-{sub_id}, run={run}."
        )

    missing_this_run = int(merged[y_col].isna().sum())

    if strict and missing_this_run > 0:
        raise ValueError(
            f'[{sub_id}] run {run} has {missing_this_run} missing {y_col} values '
            'after subjective-value merge.'
        )

    merged = merged.dropna(subset=[y_col]).reset_index(drop=True)
    return merged, missing_this_run


def build_subject_behav_bold_df(
    cfg: Any,
    sub_id: str,
    *,
    y_col: str = 'amount',
    min_runs_required: int = 3,
    min_trials_per_run: int = 20,
    sv_dirname: str = 'subjective_value_estimates',
    beta_series_subdir: str = 'beta_series',
    strict: bool = True,
    verbose: bool = True,
) -> SubjectBehavBoldResult:
    """
    Build a per-subject trial table (behavior × beta-series file paths).

    Steps:
      - load initial QA subject×run list and high-VIF omissions table
      - determine the subject's QA-passed runs
      - for each run:
          - load design matrix / behavioral trial table via make_design_matrix
          - keep only trial regressors (trial_type matching '^trial')
          - drop high-VIF trials and trials with missing beta files
          - merge behavior with available betas
      - enforce minimum usable runs and minimum trials per run

    Parameters
    ----------
    cfg
        Config object with paths (output_root, subject_lists, data_root).
    sub_id
        Subject identifier.
    y_col
        Target variable to decode. If this is one of the subjective-value targets
        (`SV_LL`, `SV_chosen`, `SV_unchosen`, `SV_SS`, `DV`), subjective-value
        derivative files are merged in by run and onset.
    min_runs_required
        Require at least this many usable runs.
    min_trials_per_run
        Require at least this many trials per run after omissions.
    sv_dirname
        Directory under cfg.output_root where per-run subjective-value files live.
    strict
        If True, raise ValueError when requirements are not met; otherwise warn.
    verbose
        If True, print run lists and omission counts.

    Returns
    -------
    SubjectBehavBoldResult
        Contains the concatenated behav_bold_df and QC counts.

    Raises
    ------
    ValueError
        If strict=True and the subject fails run/trial thresholds.
    """
    initial_sub_run_df, high_vif_sub_run_df = load_initial_and_vif_tables(cfg)
    high_vif_trials = make_high_vif_trial_set(high_vif_sub_run_df)

    runs = get_subject_good_runs(initial_sub_run_df, sub_id)
    if verbose:
        print(f'[{sub_id}] QA-passed runs: {runs}')

    if len(runs) < min_runs_required:
        msg = (
            f'[{sub_id}] has only {len(runs)} QA-passed runs; '
            f'requires >= {min_runs_required}.'
        )
        if strict:
            raise ValueError(msg)
        if verbose:
            print('WARNING:', msg)

    output_dir = (
        Path(cfg.output_root)
        / beta_series_subdir
        / 'first_level'
        / f'sub-{sub_id}'
        / 'contrast_estimates'
    )

    behav_bold_all: list[pd.DataFrame] = []
    n_trials_before = 0
    n_missing_betas = 0
    n_high_vif_omitted = 0
    n_missing_sv_rows = 0
    kept_by_run: dict[str, int] = {}

    for run in runs:
        behav_data, _, _ = make_design_matrix(cfg, sub_id, run)

        trial_mask = behav_data['trial_type'].astype(str).str.contains(r'^trial')
        behav_trials = behav_data.loc[trial_mask].copy()
        trial_types = behav_trials['trial_type'].astype(str).tolist()
        n_trials_before += len(trial_types)

        rows: list[dict[str, str]] = []
        for trial_type in trial_types:
            key = (str(sub_id), str(run), str(trial_type))
            beta_file = (
                output_dir
                / f'sub-{sub_id}_ses-scan1_task-itc_run-{run}_contrast-{trial_type}_output-effectsize.nii.gz'
            )

            if key in high_vif_trials:
                n_high_vif_omitted += 1
                continue
            if not beta_file.exists():
                n_missing_betas += 1
                continue

            rows.append(
                {'trial_type': trial_type, 'beta_file': str(beta_file), 'run': str(run)}
            )

        if not rows:
            kept_by_run[str(run)] = 0
            continue

        bold_df = pd.DataFrame(rows)

        behav_bold_run = behav_trials.merge(
            bold_df, on='trial_type', how='inner', validate='one_to_one'
        )

        if _is_sv_target(y_col):
            behav_bold_run, missing_this_run = _merge_subjective_value_columns(
                behav_bold_run,
                cfg,
                sub_id,
                run,
                y_col=y_col,
                sv_dirname=sv_dirname,
                strict=strict,
                verbose=verbose,
            )
            n_missing_sv_rows += missing_this_run

        kept_by_run[str(run)] = int(len(behav_bold_run))
        behav_bold_all.append(behav_bold_run)

    behav_bold_df = (
        pd.concat(behav_bold_all, ignore_index=True)
        if behav_bold_all
        else pd.DataFrame()
    )

    passing_runs = [r for r, n in kept_by_run.items() if n >= min_trials_per_run]

    if verbose:
        print(f'[{sub_id}] trial regressors in design (pre-filter): {n_trials_before}')
        print(f'[{sub_id}] omitted high-VIF trials: {n_high_vif_omitted}')
        print(f'[{sub_id}] missing beta files: {n_missing_betas}')
        if _is_sv_target(y_col):
            print(f'[{sub_id}] missing subjective-value rows after merge: {n_missing_sv_rows}')
        print(f'[{sub_id}] kept trials (post-filter): {len(behav_bold_df)}')
        print(f'[{sub_id}] kept by run: {kept_by_run}')
        print(
            f'[{sub_id}] runs with >= {min_trials_per_run} kept trials: {passing_runs}'
        )

    if len(passing_runs) < min_runs_required:
        msg = (
            f'[{sub_id}] only {len(passing_runs)} runs have >= {min_trials_per_run} '
            f'kept trials; requires >= {min_runs_required}. kept_by_run={kept_by_run}'
        )
        if strict:
            raise ValueError(msg)
        if verbose:
            print('WARNING:', msg)

    # Drop low-trial runs if we still have enough remaining
    if len(passing_runs) >= min_runs_required and len(passing_runs) < len(runs):
        if verbose:
            print(
                f'[{sub_id}] dropping low-trial runs: {sorted(set(runs) - set(passing_runs))}'
            )
        behav_bold_df = behav_bold_df.loc[
            behav_bold_df['run'].astype(str).isin(passing_runs)
        ].reset_index(drop=True)
        runs = passing_runs
        kept_by_run = {r: kept_by_run[r] for r in passing_runs}

    return SubjectBehavBoldResult(
        behav_bold_df=behav_bold_df,
        good_runs=[str(r) for r in runs],
        n_trials_before_vif=int(n_trials_before),
        n_trials_after_vif_and_missing=int(len(behav_bold_df)),
        n_missing_betas=int(n_missing_betas),
        n_high_vif_omitted=int(n_high_vif_omitted),
        n_missing_sv_rows=int(n_missing_sv_rows),
        trials_kept_by_run=kept_by_run,
        runs_passing_trial_threshold=passing_runs,
    )
