from typing import Dict, List

import pandas as pd

from dd_kable_analysis.config_loader import Config


def make_accept_reject_contrast_string(behav_data: pd.DataFrame) -> str:
    """
    Create a contrast string for accept vs. reject trials based on trial_type labels,
    omitting trials with missing responses ('no_response').

    Parameters
    ----------
    behav_data : pd.DataFrame
        Behavioral data with columns 'choseAccept' (0/1) and 'trial_type'.

    Returns
    -------
    str
        Contrast string suitable for Nilearn (e.g., '1/3*trial01 + 1/3*trial03 - 1/2*trial02').
    """
    # Keep only trials with valid responses
    valid_trials = behav_data.loc[behav_data['trial_type'] != 'no_response'].copy()

    accept_trials: List[str] = valid_trials.loc[
        valid_trials['choseAccept'] == 1, 'trial_type'
    ].tolist()
    reject_trials: List[str] = valid_trials.loc[
        valid_trials['choseAccept'] == 0, 'trial_type'
    ].tolist()

    n_accept = len(accept_trials)
    n_reject = len(reject_trials)

    if n_accept == 0 or n_reject == 0:
        raise ValueError('Both accept and reject trials are required for contrast.')

    pos_weight = f'1/{n_accept}'
    neg_weight = f'1/{n_reject}'

    pos_terms = [f'{pos_weight}*{t}' for t in accept_trials]
    neg_terms = [f'-{neg_weight}*{t}' for t in reject_trials]

    return ' + '.join(pos_terms + neg_terms)


def make_beta_series_constrast_set(
    desmat: pd.DataFrame, behav_data: pd.DataFrame
) -> Dict[str, str]:
    """
    Create the set of contrasts for the beta series model.  Generates a contrast for each
      trial's beta as well as the accept - reject contrast.

    Parameters
    ----------
    desmat: pd.DataFrame
        The design matrix that will be used in the model
    behav_data: pd.DataFrame
        The partially processed behavioral data that are output by make_design_matrix()

    Returns
    -------
    contrasts: Dict[str, str]
        A dictionary of contrasts that are nilearn friendly when used with desmat.
    """
    if 'trial_type' not in behav_data.columns:
        raise ValueError('Input `behav_data` must be output from make_design_matrix()')

    contrasts = {
        col_name: col_name for col_name in desmat.columns if 'trial' in col_name
    }

    accept_reject_contrast = make_accept_reject_contrast_string(behav_data)
    contrasts['accept_minus_reject'] = accept_reject_contrast
    return contrasts
