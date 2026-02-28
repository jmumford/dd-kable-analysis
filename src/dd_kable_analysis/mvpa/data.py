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
    trials_kept_by_run: dict[str, int] | None = None
    runs_passing_trial_threshold: list[str] | None = None


def build_subject_behav_bold_df(
    cfg: Any,
    sub_id: str,
    *,
    min_runs_required: int = 3,
    min_trials_per_run: int = 20,
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
    min_runs_required
        Require at least this many usable runs.
    min_trials_per_run
        Require at least this many trials per run after omissions.
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
        / 'beta_series'
        / 'first_level'
        / f'sub-{sub_id}'
        / 'contrast_estimates'
    )

    behav_bold_all: list[pd.DataFrame] = []
    n_trials_before = 0
    n_missing_betas = 0
    n_high_vif_omitted = 0
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
        trials_kept_by_run=kept_by_run,
        runs_passing_trial_threshold=passing_runs,
    )
