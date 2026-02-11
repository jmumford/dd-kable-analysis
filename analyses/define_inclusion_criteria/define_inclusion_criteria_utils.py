import re
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from dd_kable_analysis.data_io import (
    load_tsv_data,
    resolve_file,
)


# This function is a bit fragile, since it is parsing text that
# wasn't written in a consistent fashion (has been double checked)
def parse_mri_notes_to_exclusions(
    inclusion_subjects: pd.DataFrame,
    notes_col: str = 'MRI Notes',
    sub_col: str = 'sub_id',
    default_runs: Tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """
    Parse the MRI notes column in inclusion_subjects and return a tidy dataframe
    of MRI QA exclusions, one row per run to exclude.

    Parameters
    ----------
    inclusion_subjects : pd.DataFrame
        DataFrame with subject info and notes.
    notes_col : str
        Column containing MRI notes.
    sub_col : str
        Column containing subject IDs.
    default_runs : tuple[int]
        Runs to consider for inclusion.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['sub_id', 'run', 'reason'].
    """
    pattern = re.compile(
        r'(?is)\bRun(?:s)?\s*'
        r'(?P<runs>\d+(?:\s*[-,]\s*\d+)*)\s*'
        r':\s*(?P<reason>[^;\n]+)\s*;?'
    )

    def expand_runs(runs_text: str) -> List[int]:
        runs_text = runs_text.strip()
        out: List[int] = []
        for part in [p.strip() for p in runs_text.split(',')]:
            if '-' in part:
                a, b = [int(x.strip()) for x in part.split('-')]
                out.extend(range(min(a, b), max(a, b) + 1))
            else:
                out.append(int(part))
        return out

    rows: List[dict] = []
    for _, row in inclusion_subjects.iterrows():
        sub_id = row[sub_col]
        notes = row.get(notes_col)
        if pd.isna(notes) or str(notes).strip() == '':
            continue

        txt = str(notes).replace('\r\n', '\n')
        for match in pattern.finditer(txt):
            runs = expand_runs(match.group('runs'))
            reason = ' '.join(match.group('reason').split())
            for run_num in runs:
                if run_num in default_runs:
                    rows.append({'sub_id': sub_id, 'run': run_num, 'reason': reason})

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(['sub_id', 'run', 'reason'])
        .reset_index(drop=True)
    )


def check_mri_qa_exclusion(
    mri_qa_exclusions: pd.DataFrame, sub_id: str, run: int
) -> str | None:
    """
    Return MRI QA exclusion reason for a given subject and run, or None.
    """
    hits = mri_qa_exclusions.loc[
        (mri_qa_exclusions['sub_id'] == sub_id)
        & (mri_qa_exclusions['run'].astype(int) == run),
        'reason',
    ]
    if hits.empty:
        return None
    return '; '.join(pd.unique(hits.astype(str)))


def check_bold_exists(cfg: Any, sub_id: str, session: str, run: int) -> bool:
    """
    Return True if the fMRIPrep BOLD file exists for this (sub, session, run),
    otherwise False.
    """
    try:
        bold_file = resolve_file(cfg, sub_id, session, run, 'bold')
    except ValueError:
        return False
    return Path(bold_file).exists()


def evaluate_behav_run(behav_data: pd.DataFrame) -> Tuple[bool, bool, bool]:
    """
    Evaluate a behavioral run for missing RTs and extreme chooseAccept proportions.

    Returns
    -------
    Tuple[no_data, gt_10_missing, chose_accept_lt10_or_gt90]
    """
    if behav_data is None or behav_data.empty:
        return True, False, False

    prop_missing_rt = behav_data['RT'].isna().mean()
    ca_nonmiss = pd.to_numeric(behav_data['choseAccept'], errors='coerce').dropna()
    prop_accept = np.nan if ca_nonmiss.empty else ca_nonmiss.mean()

    return (
        False,
        prop_missing_rt > 0.10,
        np.isnan(prop_accept) or (prop_accept < 0.10) or (prop_accept > 0.90),
    )


def make_good_bad_runs(
    cfg: Any, inclusion_subjects: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames of good and bad runs for all subjects.

    Flags for bad runs:
        - Behavioral data missing or bad
        - MRI QA exclusion
        - BOLD file missing
        - Fewer than 2 good runs per subject

    Returns
    -------
    good_data : pd.DataFrame
        Rows corresponding to runs passing all checks.
    bad_data : pd.DataFrame
        Rows corresponding to runs failing any check, with reason flags.
    """
    good_rows, bad_rows = [], []
    session = 'scan1'
    mri_qa_exclusions = parse_mri_notes_to_exclusions(inclusion_subjects)

    for sub_id in inclusion_subjects['sub_id']:
        for run_str in ['1', '2', '3', '4']:
            run = int(run_str)
            row_base = {'sub_id': sub_id, 'session': session, 'run': run}

            # Initialize flags
            no_behav_data = False
            gt_10_missing = False
            chose_accept_lt10_or_gt90 = False
            no_fmriprep_bold = not check_bold_exists(cfg, sub_id, session, run)
            mri_qa_reason = check_mri_qa_exclusion(mri_qa_exclusions, sub_id, run)

            # Behavioral check
            try:
                behav_file = resolve_file(cfg, sub_id, session, run, 'behav')
                behav_data = load_tsv_data(behav_file)
                no_behav_data, gt_10_missing, chose_accept_lt10_or_gt90 = (
                    evaluate_behav_run(behav_data)
                )
            except ValueError:
                no_behav_data = True

            is_bad = any(
                [
                    no_behav_data,
                    gt_10_missing,
                    chose_accept_lt10_or_gt90,
                    no_fmriprep_bold,
                    mri_qa_reason is not None,
                ]
            )

            if is_bad:
                bad_rows.append(
                    {
                        **row_base,
                        'gt_10_missing': gt_10_missing,
                        'chose_accept_lt10_or_gt90': chose_accept_lt10_or_gt90,
                        'no_behav_data': no_behav_data,
                        'no_fmriprep_bold': no_fmriprep_bold,
                        'fewer_than_2_good_runs': False,
                        'mri_qa_exclusion': mri_qa_reason,
                    }
                )
            else:
                good_rows.append(row_base)

    good_data = pd.DataFrame(good_rows, columns=['sub_id', 'session', 'run'])
    bad_data = pd.DataFrame(
        bad_rows,
        columns=[
            'sub_id',
            'session',
            'run',
            'gt_10_missing',
            'chose_accept_lt10_or_gt90',
            'no_behav_data',
            'no_fmriprep_bold',
            'fewer_than_2_good_runs',
            'mri_qa_exclusion',
        ],
    )

    # Subject-level rule: omit subjects with <2 good runs
    good_counts = (
        good_data.groupby('sub_id').size() if len(good_data) else pd.Series(dtype=int)
    )
    bad_subjects = set(good_counts[good_counts < 2].index)

    if bad_subjects:
        to_move = good_data[good_data['sub_id'].isin(bad_subjects)].copy()
        good_data = good_data[~good_data['sub_id'].isin(bad_subjects)].copy()

        moved_bad = to_move.assign(
            gt_10_missing=False,
            chose_accept_lt10_or_gt90=False,
            no_behav_data=False,
            no_fmriprep_bold=False,
            fewer_than_2_good_runs=True,
            mri_qa_exclusion=None,
        )

        bad_data = pd.concat([bad_data, moved_bad], ignore_index=True)
        bad_data.loc[
            bad_data['sub_id'].isin(bad_subjects), 'fewer_than_2_good_runs'
        ] = True

    return (
        good_data.sort_values(['sub_id', 'session', 'run']).reset_index(drop=True),
        bad_data.sort_values(['sub_id', 'session', 'run']).reset_index(drop=True),
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLD is below here
def parse_mri_notes_to_exclusions_OLD(
    inclusion_subjects: pd.DataFrame,
    notes_col='MRI Notes',
    sub_col='sub_id',
    default_runs=(1, 2, 3, 4),
) -> pd.DataFrame:
    """
    Returns a tidy dataframe: sub_id, run, reason
    One row per run to exclude, parsed from inclusion_subjects[notes_col].
    """

    # Matches:
    # Run1: ...;
    # Run 1: ...;
    # Runs 1-4: ...;
    # Runs 1,3: ...;
    patt = re.compile(
        r'(?is)\bRun(?:s)?\s*'  # Run / Runs
        r'(?P<runs>\d+(?:\s*[-,]\s*\d+)*)\s*'  # 1  or  1-4  or 1,3
        r':\s*(?P<reason>[^;\n]+)\s*;?'  # reason up to ; or newline
    )

    def expand_runs(runs_text: str):
        runs_text = runs_text.strip()
        out = []
        # split on commas first
        parts = [p.strip() for p in runs_text.split(',')]
        for p in parts:
            if '-' in p:
                a, b = [int(x.strip()) for x in p.split('-')]
                out.extend(list(range(min(a, b), max(a, b) + 1)))
            else:
                out.append(int(p))
        return out

    rows = []
    for _, row in inclusion_subjects.iterrows():
        sub_id = row[sub_col]
        notes = row.get(notes_col)

        if pd.isna(notes) or str(notes).strip() == '':
            continue

        txt = str(notes).replace('\r\n', '\n')

        matches = list(patt.finditer(txt))
        if not matches:
            continue

        for m in matches:
            runs = expand_runs(m.group('runs'))
            reason = ' '.join(m.group('reason').split())  # normalize whitespace

            for rnum in runs:
                if rnum in default_runs:
                    rows.append({'sub_id': sub_id, 'run': rnum, 'reason': reason})

    mri_qa_exclusions = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(['sub_id', 'run', 'reason'])
        .reset_index(drop=True)
    )
    return mri_qa_exclusions


def check_mri_qa_exclusion_OLD(mri_qa_exclusions: pd.DataFrame, sub_id, run):
    """
    Returns the MRI QA exclusion reason string for (sub_id, run), else None.
    session is intentionally ignored.
    """
    run = int(run)
    hits = mri_qa_exclusions.loc[
        (mri_qa_exclusions['sub_id'] == sub_id)
        & (mri_qa_exclusions['run'].astype(int) == run),
        'reason',
    ]
    if hits.empty:
        return None
    # if multiple reasons exist, join them
    return '; '.join(pd.unique(hits.astype(str)))


def check_bold_exists_OLD(cfg, sub_id, session, run):
    """
    Return True if the fMRIPrep BOLD file exists for this (sub, session, run),
    otherwise False.
    """
    try:
        bold_file = resolve_file(cfg, sub_id, session, run, 'bold')
    except ValueError:
        return False

    return Path(bold_file).exists()


def evaluate_behav_run_OLD(behav_data: pd.DataFrame):
    if behav_data is None or behav_data.empty:
        return True, False, False

    prop_missing_rt = behav_data['RT'].isna().mean()

    ca_nonmiss = pd.to_numeric(behav_data['choseAccept'], errors='coerce').dropna()
    prop_accept = np.nan if ca_nonmiss.empty else ca_nonmiss.mean()

    gt_10_missing = prop_missing_rt > 0.10
    chose_accept_lt10_or_gt90 = (
        np.isnan(prop_accept) or (prop_accept < 0.10) or (prop_accept > 0.90)
    )

    return False, gt_10_missing, chose_accept_lt10_or_gt90


def make_good_bad_runs_OLD(cfg, inclusion_subjects: pd.DataFrame):
    good_rows, bad_rows = [], []
    session = 'scan1'
    mri_qa_exclusions = parse_mri_notes_to_exclusions(inclusion_subjects)

    for sub_id in inclusion_subjects['sub_id']:
        for run in ['1', '2', '3', '4']:
            row_base = {
                'sub_id': sub_id,
                'session': session,
                'run': run,
            }

            # Defaults
            no_behav_data = False
            gt_10_missing = False
            chose_accept_lt10_or_gt90 = False
            no_fmriprep_bold = False

            # MRI QA flag (reason text or None)
            mri_qa_reason = check_mri_qa_exclusion(mri_qa_exclusions, sub_id, run)

            # Check BOLD existence
            no_fmriprep_bold = not check_bold_exists(cfg, sub_id, session, run)

            # Behavioral checks
            try:
                behav_file = resolve_file(cfg, sub_id, session, run, 'behav')
                behav_data = load_tsv_data(behav_file)
                no_behav_data, gt_10_missing, chose_accept_lt10_or_gt90 = (
                    evaluate_behav_run(behav_data)
                )
            except ValueError:
                no_behav_data = True

            is_bad = (
                no_behav_data
                or gt_10_missing
                or chose_accept_lt10_or_gt90
                or no_fmriprep_bold
                or (mri_qa_reason is not None)
            )

            if is_bad:
                bad_rows.append(
                    {
                        **row_base,
                        'gt_10_missing': bool(gt_10_missing),
                        'chose_accept_lt10_or_gt90': bool(chose_accept_lt10_or_gt90),
                        'no_behav_data': bool(no_behav_data),
                        'no_fmriprep_bold': bool(no_fmriprep_bold),
                        'fewer_than_2_good_runs': False,
                        'mri_qa_exclusion': mri_qa_reason,
                    }
                )
            else:
                good_rows.append(row_base)

    good_data = pd.DataFrame(good_rows, columns=['sub_id', 'session', 'run'])

    bad_data = pd.DataFrame(
        bad_rows,
        columns=[
            'sub_id',
            'session',
            'run',
            'gt_10_missing',
            'chose_accept_lt10_or_gt90',
            'no_behav_data',
            'no_fmriprep_bold',
            'fewer_than_2_good_runs',
            'mri_qa_exclusion',
        ],
    )

    # Subject-level rule: omit subject if <2 good runs
    good_counts = (
        good_data.groupby('sub_id').size() if len(good_data) else pd.Series(dtype=int)
    )
    bad_subjects = set(good_counts[good_counts < 2].index)

    if bad_subjects:
        to_move = good_data[good_data['sub_id'].isin(bad_subjects)].copy()
        good_data = good_data[~good_data['sub_id'].isin(bad_subjects)].copy()

        moved_bad = to_move.assign(
            gt_10_missing=False,
            chose_accept_lt10_or_gt90=False,
            no_behav_data=False,
            no_fmriprep_bold=False,
            fewer_than_2_good_runs=True,
            mri_qa_exclusion=None,
        )

        bad_data = pd.concat([bad_data, moved_bad], ignore_index=True)
        bad_data.loc[
            bad_data['sub_id'].isin(bad_subjects), 'fewer_than_2_good_runs'
        ] = True

    return (
        good_data.sort_values(['sub_id', 'session', 'run']).reset_index(drop=True),
        bad_data.sort_values(['sub_id', 'session', 'run']).reset_index(drop=True),
    )
