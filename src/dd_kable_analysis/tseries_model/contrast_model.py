from typing import Dict, List

import pandas as pd

from dd_kable_analysis.config_loader import Config


def make_ll_minus_ss_contrast_string(behav_data: pd.DataFrame) -> str:
    """
    Create a contrast string for LL minus SS trials based on trial_type labels,
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
            trial's beta as well as the ll_minus_ss contrast.

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

    ll_minus_ss_contrast = make_ll_minus_ss_contrast_string(behav_data)
    contrasts['ll_minus_ss'] = ll_minus_ss_contrast
    return contrasts


def make_traditional_contrast_set(desmat: pd.DataFrame) -> Dict[str, str]:
    """
    Create the set of contrasts for the traditional model (LL minus SS).

    Parameters
    ----------
    desmat: pd.DataFrame
        The design matrix used in the model (expects ll/ss columns).

    Returns
    -------
    Dict[str, str]
        A dictionary of contrasts for nilearn.
    """
    return {'ll_minus_ss': 'll - ss'}
