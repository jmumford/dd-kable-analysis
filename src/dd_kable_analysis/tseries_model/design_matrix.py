import re
from pathlib import Path
from typing import Any, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from dd_kable_analysis.config_loader import Config
from dd_kable_analysis.data_io import (
    load_tsv_data,
    resolve_file,
)


def get_confounds(
    cfg: Config,
    sub_id: str,
    run: Union[int, str],
) -> pd.DataFrame:
    """
    Load and select nuisance confound regressors for a given subject and run.

    Parameters
    ----------
    cfg : Config
        Analysis configuration object.
    sub_id : str
        Subject identifier.
    run : int or str
        Run identifier.

    Returns
    -------
    pd.DataFrame
        Confound regressors with selected columns and NaNs filled with zeros.
    """
    confounds_file = resolve_file(cfg, sub_id, 'scan1', run, 'confounds')
    confounds = load_tsv_data(confounds_file)

    patterns = cfg.confounds.compiled_patterns

    cols_to_keep = [
        col for col in confounds.columns if any(p.match(col) for p in patterns)
    ]

    return confounds[cols_to_keep].fillna(0)


def get_frametimes(cfg: Config, sub_id: str, run: str) -> np.ndarray:
    """
    Compute frame times for a BOLD image based on TR.

    Parameters
    ----------
    cfg : Config
        Analysis configuration object containing TR and other settings.

    Returns
    -------
    np.ndarray
        Array of frame times (seconds) with length equal to number of time points.
    """
    bold_file = resolve_file(cfg, sub_id, 'scan1', run, 'bold')
    hdr = nib.Nifti1Image.from_filename(bold_file).header
    n_timepoints = max(1, hdr['dim'][4])

    TR = float(cfg.tr)

    return np.arange(n_timepoints) * TR


def make_design_matrix(
    cfg: Config, sub_id: str, run: Union[int, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a Nilearn first-level design matrix with:
      - One regressor per trial (for trials with valid RT)
      - Single RT regressor
      - 'no_response' regressor for trials with missing RT
      - Confound regressors
      - Intercept

    Parameters
    ----------
    cfg : Config
        Analysis configuration object containing TR and other settings.
    sub_id : str
        Subject identifier.
    run : int or str
        Run number.

    Returns
    -------
    pd.DataFrame
        behavioral data
    pd.DataFrame
        Events data used to make convolved regressors
    pd.DataFrame
        First-level design matrix.
    """
    # --- Frame times and confounds ---
    frame_times = get_frametimes(cfg, sub_id, run)
    confounds = get_confounds(cfg, sub_id, run)

    # --- Load behavioral data ---
    behav_file = resolve_file(cfg, sub_id, 'scan1', run, 'behav')
    behav_data = load_tsv_data(behav_file)

    # Drop the last trial (max onset): too close to the end for all runs/subjects.
    behav_data = behav_data.loc[behav_data['onset'] != behav_data['onset'].max()].reset_index(drop=True)

    # --- Check for clipped trials ---
    clipped_mask = np.ceil(behav_data['onset']) >= frame_times[-1]
    n_clipped = clipped_mask.sum()
    if n_clipped > 0:
        msg = f'❌ {n_clipped} trial{"s" if n_clipped > 1 else ""} for sub {sub_id}, run {run} are clipped.'
        if n_clipped == 1:
            print(msg + ' Removing the trial and continuing.')
            behav_data = behav_data.loc[~clipped_mask].reset_index(drop=True)
        else:
            # Raise a short error message instead of long traceback
            raise RuntimeError(msg + ' Cannot generate design matrix.')

    # Remove clipped trials
    behav_data = behav_data.loc[~clipped_mask].reset_index(drop=True)

    # --- Number trials by row order ---
    behav_data = behav_data.sort_values('onset').reset_index(drop=True)
    behav_data['trial_num'] = behav_data.index

    # --- Trial type for Nilearn ---
    behav_data['trial_type'] = [f'trial{i:02d}' for i in behav_data['trial_num']]
    rt_missing_mask = behav_data['RT'].isna()
    behav_data.loc[rt_missing_mask, 'trial_type'] = 'no_response'

    events_model_rt = behav_data.loc[~rt_missing_mask, ['onset', 'RT']].copy()
    events_model_rt['trial_type'] = 'rt'
    events_model_rt.rename(columns={'RT': 'duration'}, inplace=True)

    # --- Build events DataFrame ---
    events_model = pd.concat(
        [
            # Trial-specific regressors
            behav_data[['onset', 'duration', 'trial_type']],
            # RT parametric regressor
            events_model_rt,
        ],
        ignore_index=True,
    )

    # --- Make design matrix ---
    desmat = make_first_level_design_matrix(
        frame_times,
        events=events_model,
        hrf_model='spm',
        drift_model=None,
        add_regs=confounds.values,
        add_reg_names=confounds.columns.tolist(),
    )

    return behav_data, events_model, desmat


def make_design_matrix_traditional(
    cfg: Config, sub_id: str, run: Union[int, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Build a Nilearn first-level design matrix with:
            - LL regressor
            - SS regressor
      - RT regressor (RT modeled as duration)
      - 'no_response' regressor for trials with missing RT
      - Confound regressors
      - Intercept

    Parameters
    ----------
    cfg : Config
        Analysis configuration object containing TR and other settings.
    sub_id : str
        Subject identifier.
    run : int or str
        Run number.

    Returns
    -------
    pd.DataFrame
        behavioral data
    pd.DataFrame
        Events data used to make convolved regressors
    pd.DataFrame
        First-level design matrix.
    """
    # --- Frame times and confounds ---
    frame_times = get_frametimes(cfg, sub_id, run)
    confounds = get_confounds(cfg, sub_id, run)

    # --- Load behavioral data ---
    behav_file = resolve_file(cfg, sub_id, 'scan1', run, 'behav')
    behav_data = load_tsv_data(behav_file)

    # --- Check for clipped trials ---
    clipped_mask = np.ceil(behav_data['onset']) >= frame_times[-1]
    n_clipped = clipped_mask.sum()
    if n_clipped > 0:
        msg = f'❌ {n_clipped} trial{"s" if n_clipped > 1 else ""} for sub {sub_id}, run {run} are clipped.'
        if n_clipped == 1:
            print(msg + ' Removing the trial and continuing.')
            behav_data = behav_data.loc[~clipped_mask].reset_index(drop=True)
        else:
            raise RuntimeError(msg + ' Cannot generate design matrix.')

    # Remove clipped trials
    behav_data = behav_data.loc[~clipped_mask].reset_index(drop=True)

    # --- Trial type for traditional regressors ---
    rt_missing_mask = behav_data['RT'].isna()

    events_accept_reject = behav_data.loc[
        ~rt_missing_mask, ['onset', 'duration', 'choseAccept']
    ].copy()
    events_accept_reject['trial_type'] = np.where(
        events_accept_reject['choseAccept'] == 1, 'll', 'ss'
    )
    events_accept_reject = events_accept_reject.drop(columns=['choseAccept'])

    events_no_response = behav_data.loc[rt_missing_mask, ['onset', 'duration']].copy()
    events_no_response['trial_type'] = 'no_response'

    events_model_rt = behav_data.loc[~rt_missing_mask, ['onset', 'RT']].copy()
    events_model_rt['trial_type'] = 'rt'
    events_model_rt.rename(columns={'RT': 'duration'}, inplace=True)

    # --- Build events DataFrame ---
    events_model = pd.concat(
        [events_accept_reject, events_no_response, events_model_rt],
        ignore_index=True,
    )

    # --- Make design matrix ---
    desmat = make_first_level_design_matrix(
        frame_times,
        events=events_model,
        hrf_model='spm',
        drift_model=None,
        add_regs=confounds.values,
        add_reg_names=confounds.columns.tolist(),
    )

    return behav_data, events_model, desmat
