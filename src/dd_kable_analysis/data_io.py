import os
import re
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from dd_kable_analysis.config_loader import Config


def resolve_file(cfg: Config, subject_id: str, ses: str, run: str, kind: str) -> Path:
    """
    Fetch a single file for a subject based on kind ('bold', 'mask', or 'behav').
    Ensures exactly one file is matched and returns it as a Path.
    """

    if kind == 'bold':
        pattern = f'{cfg.fmriprep_dir}/sub-{subject_id}/' + cfg.bold_file_glob.format(
            subject=subject_id, ses=ses, run=run
        )

    elif kind == 'mask':
        pattern = (
            f'{cfg.fmriprep_dir}/sub-{subject_id}/'
            + cfg.bold_mask_file_glob.format(subject=subject_id, ses=ses, run=run)
        )

    elif kind == 'behav':
        pattern = f'{cfg.bids_dir}/sub-{subject_id}/' + cfg.behav_file_glob.format(
            subject=subject_id, ses=ses, run=run
        )

    elif kind == 'confounds':
        pattern = (
            f'{cfg.fmriprep_dir}/sub-{subject_id}/'
            + cfg.confounds_file_glob.format(subject=subject_id, ses=ses, run=run)
        )

    else:
        raise ValueError(
            f'Unknown file kind: {kind!r}. Must be one of '
            f"'bold', 'mask', 'behav', 'confounds'."
        )

    matches = glob(pattern)

    if not matches:
        raise ValueError(f'No files found for {kind!r} with pattern: {pattern}')
    if len(matches) > 1:
        raise ValueError(
            f'Multiple files found for {kind!r} with pattern: {pattern}\n{matches}'
        )

    return Path(matches[0])


def load_tsv_data(tsv_file: Path) -> pd.DataFrame:
    """
    Load behavioral data from a TSV file into a pandas DataFrame.

    Parameters
    ----------
    tsv_file : Path
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        Data from the TSV file.

    Raises
    ------
    ValueError
        If the file does not have a .tsv extension.
    """
    if tsv_file.suffix != '.tsv':
        raise ValueError(f'Expected a .tsv file, got: {tsv_file}')

    return pd.read_csv(tsv_file, sep='\t')


def get_subids_bids_dir(cfg: Config) -> List[str]:
    """
    Extract unique subject IDs from cfg.bids_dir.

    Looks for directories starting with 'sub-...' and returns the
    string after the 'sub-' prefix (e.g. 'dmp0005').
    """
    subids: set[str] = set()
    bids_dir = Path(cfg.bids_dir)

    if not bids_dir.exists() or not bids_dir.is_dir():
        raise ValueError(f'bids_dir does not exist or is not a directory: {bids_dir}')

    pattern = re.compile(r'^sub-(.+)$')  # capture everything after "sub-"

    for entry in bids_dir.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                subids.add(m.group(1))

    return sorted(subids)


def get_confounds_data(confounds_file):
    """
    Creates nuisance regressors for output from fmriprep.
    input:
        confounds_file: path to confounds file from fmriprep
    output:
        confound_regressors: includes motion regressors, FD, nonsteady volumes,
            cosine basis set
        percent_high_motion:  Percentage of high motion time points.  High motion
            is defined by the following
            FD>.5, stdDVARS>1.2 (that relates to DVARS>.5)
    """
    confounds_df = pd.read_csv(confounds_file, sep='\t', na_values=['n/a']).fillna(0)
    excessive_movement = (confounds_df.framewise_displacement > 0.5) | (
        confounds_df.std_dvars > 1.2
    )
    percent_high_motion = np.mean(excessive_movement)
    confounds = confounds_df.filter(
        regex='^(cosine00|cosine01|cosine02|trans|rot)'
    ).copy()
    confounds = confounds.loc[:, ~confounds.columns.str.endswith('power2')]
    return confounds, percent_high_motion
