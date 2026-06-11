import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

SS_AMOUNT = 20.0


def subjective_value(amount, delay, k):
    """Hyperbolic subjective value of a delayed reward."""
    return amount / (1 + k * delay)


def compute_trialwise_measures(
    df_sub,
    k_hat,
    *,
    amount_col='amount',
    delay_col='Delay',
    choice_col='choseAccept',
):
    """Add trial-wise subjective-value measures for one fitted subject.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Trial-level data for one subject.
    k_hat : float
        Fitted discount-rate parameter for the subject.
    amount_col, delay_col, choice_col : str
        Column names for amount, delay, and binary LL choice indicator.
    """
    temp = df_sub.copy()

    sv_ll = subjective_value(
        temp[amount_col].to_numpy(dtype=float),
        temp[delay_col].to_numpy(dtype=float),
        float(k_hat),
    )
    sv_ss = np.full(len(temp), SS_AMOUNT, dtype=float)
    dv = sv_ll - sv_ss

    chose_ll = temp[choice_col].to_numpy().astype(bool)
    sv_chosen = np.where(chose_ll, sv_ll, sv_ss)
    sv_unchosen = np.where(chose_ll, sv_ss, sv_ll)

    temp['SV_LL'] = sv_ll
    temp['SV_SS'] = sv_ss
    temp['DV'] = dv
    temp['SV_chosen'] = sv_chosen
    temp['SV_unchosen'] = sv_unchosen

    return temp


def neg_log_likelihood(params, amount, delay, choices):
    """Negative log likelihood for the hyperbolic discounting choice model.

    Parameters
    ----------
    params : array-like of shape (2,)
        ``[log_k, log_beta]``.
    amount : array-like
        Delayed-larger reward amounts.
    delay : array-like
        Delays for delayed-larger rewards.
    choices : array-like
        Binary choices where 1 indicates LL and 0 indicates SS.
    """
    log_k, log_beta = params

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        k = np.exp(log_k)
        beta = np.exp(log_beta)

    if not np.isfinite(k) or not np.isfinite(beta):
        return np.inf

    sv_ll = subjective_value(amount, delay, k)
    dv = sv_ll - SS_AMOUNT
    p_choose_ll = expit(beta * dv)

    eps = 1e-10
    p_choose_ll = np.clip(p_choose_ll, eps, 1 - eps)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        loglik = choices * np.log(p_choose_ll) + (1 - choices) * np.log(1 - p_choose_ll)

    if not np.all(np.isfinite(loglik)):
        return np.inf

    return -np.sum(loglik)


def _empty_attempts_df():
    return pd.DataFrame(
        columns=[
            'k',
            'beta',
            'nll',
            'success',
            'converged',
            'init_log_k',
            'init_log_beta',
            'n_invalid_evaluations',
            'n_warnings',
            'hit_invalid_params',
        ]
    )


def _fit_failure_summary(df_sub, df_clean):
    prop_chose_ll = np.nan
    choice_std = np.nan

    if len(df_clean) > 0:
        choices = df_clean['choseAccept'].values
        prop_chose_ll = np.mean(choices)
        choice_std = np.std(choices)

    return {
        'all_attempts': _empty_attempts_df(),
        'n_converged': 0,
        'best_k': np.nan,
        'best_beta': np.nan,
        'best_nll': np.nan,
        'k_cv': np.nan,
        'beta_cv': np.nan,
        'n_invalid_evaluations_total': 0,
        'n_warnings_total': 0,
        'n_attempts_with_invalid_params': 0,
        'n_failed_initializations': 0,
        'n_trials': len(df_sub),
        'n_trials_clean': len(df_clean),
        'n_missing': len(df_sub) - len(df_clean),
        'prop_chose_LL': prop_chose_ll,
        'choice_std': choice_std,
        'always_LL': prop_chose_ll == 1.0 if np.isfinite(prop_chose_ll) else False,
        'always_SS': prop_chose_ll == 0.0 if np.isfinite(prop_chose_ll) else False,
        'nearly_constant_choice': (
            choice_std < 0.1 if np.isfinite(choice_std) else False
        ),
        'delay_min': df_clean['Delay'].min() if len(df_clean) > 0 else np.nan,
        'delay_max': df_clean['Delay'].max() if len(df_clean) > 0 else np.nan,
        'delay_mean': df_clean['Delay'].mean() if len(df_clean) > 0 else np.nan,
    }


def fit_subject_collect_all_attempts(df_sub, n_starts=20):
    """Collect all multistart optimization attempts plus data-quality diagnostics.

    This version suppresses noisy overflow warnings from exponentiating extreme
    optimizer proposals and instead treats those evaluations as invalid, which
    are counted in the returned diagnostics.
    """
    df_clean = df_sub.dropna(subset=['amount', 'Delay', 'choseAccept'])

    if len(df_clean) == 0:
        return _fit_failure_summary(df_sub, df_clean)

    amount = df_clean['amount'].values
    delay = df_clean['Delay'].values
    choices = df_clean['choseAccept'].values

    all_results = []

    for _ in range(n_starts):
        init_log_k = np.random.uniform(-6, 0)
        init_log_beta = np.random.uniform(-2, 2)
        init_params = np.array([init_log_k, init_log_beta])
        invalid_evaluations = 0
        warning_count = 0

        def objective(params):
            nonlocal invalid_evaluations
            nll = neg_log_likelihood(params, amount, delay, choices)
            if not np.isfinite(nll):
                invalid_evaluations += 1
                return np.inf
            return nll

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter('always', RuntimeWarning)
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                )

            warning_count = sum(
                issubclass(w.category, RuntimeWarning) for w in caught_warnings
            )

            k = np.exp(result.x[0]) if np.isfinite(result.x[0]) else np.nan
            beta = np.exp(result.x[1]) if np.isfinite(result.x[1]) else np.nan

            all_results.append(
                {
                    'k': k if np.isfinite(k) else np.nan,
                    'beta': beta if np.isfinite(beta) else np.nan,
                    'nll': result.fun,
                    'success': result.success,
                    'converged': result.success and np.isfinite(result.fun),
                    'init_log_k': init_params[0],
                    'init_log_beta': init_params[1],
                    'n_invalid_evaluations': invalid_evaluations,
                    'n_warnings': warning_count,
                    'hit_invalid_params': invalid_evaluations > 0,
                }
            )
        except Exception:
            all_results.append(
                {
                    'k': np.nan,
                    'beta': np.nan,
                    'nll': np.nan,
                    'success': False,
                    'converged': False,
                    'init_log_k': init_params[0],
                    'init_log_beta': init_params[1],
                    'n_invalid_evaluations': invalid_evaluations,
                    'n_warnings': warning_count,
                    'hit_invalid_params': invalid_evaluations > 0,
                }
            )

    results_df = pd.DataFrame(all_results)
    converged = results_df[results_df['converged']]

    if len(converged) > 0:
        best_idx = converged['nll'].idxmin()
        best = results_df.loc[best_idx]
        k_cv = (
            converged['k'].std() / converged['k'].mean()
            if converged['k'].mean() > 0
            else np.nan
        )
        beta_cv = (
            converged['beta'].std() / converged['beta'].mean()
            if converged['beta'].mean() > 0
            else np.nan
        )
    else:
        best = None
        k_cv = np.nan
        beta_cv = np.nan

    prop_chose_ll = np.mean(choices)
    choice_std = np.std(choices)

    return {
        'all_attempts': results_df,
        'n_converged': results_df['converged'].sum(),
        'best_k': best['k'] if best is not None else np.nan,
        'best_beta': best['beta'] if best is not None else np.nan,
        'best_nll': best['nll'] if best is not None else np.nan,
        'k_cv': k_cv,
        'beta_cv': beta_cv,
        'n_invalid_evaluations_total': int(results_df['n_invalid_evaluations'].sum()),
        'n_warnings_total': int(results_df['n_warnings'].sum()),
        'n_attempts_with_invalid_params': int(results_df['hit_invalid_params'].sum()),
        'n_failed_initializations': int((~results_df['converged']).sum()),
        'n_trials': len(df_sub),
        'n_trials_clean': len(df_clean),
        'n_missing': len(df_sub) - len(df_clean),
        'prop_chose_LL': prop_chose_ll,
        'choice_std': choice_std,
        'always_LL': prop_chose_ll == 1.0,
        'always_SS': prop_chose_ll == 0.0,
        'nearly_constant_choice': choice_std < 0.1,
        'delay_min': df_clean['Delay'].min(),
        'delay_max': df_clean['Delay'].max(),
        'delay_mean': df_clean['Delay'].mean(),
    }


def split_half_reliability(df_sub, n_starts=20):
    """Estimate split-half reliability for fitted discounting parameters."""
    df_clean = df_sub.dropna(subset=['amount', 'Delay', 'choseAccept'])

    if len(df_clean) < 2:
        return {
            'k_half1': np.nan,
            'k_half2': np.nan,
            'beta_half1': np.nan,
            'beta_half2': np.nan,
            'k_diff': np.nan,
            'beta_diff': np.nan,
            'sv_correlation': np.nan,
            'sv_mae': np.nan,
            'half1_n_invalid_evaluations': 0,
            'half2_n_invalid_evaluations': 0,
            'half1_n_warnings': 0,
            'half2_n_warnings': 0,
            'half1_n_failed_initializations': 0,
            'half2_n_failed_initializations': 0,
        }

    n_trials = len(df_clean)
    indices = np.random.permutation(n_trials)
    half1_idx = indices[: n_trials // 2]
    half2_idx = indices[n_trials // 2 :]

    df_half1 = df_clean.iloc[half1_idx].copy()
    df_half2 = df_clean.iloc[half2_idx].copy()

    results1 = fit_subject_collect_all_attempts(df_half1, n_starts=n_starts)
    results2 = fit_subject_collect_all_attempts(df_half2, n_starts=n_starts)

    k_diff = np.abs(results1['best_k'] - results2['best_k'])
    beta_diff = np.abs(results1['best_beta'] - results2['best_beta'])

    if np.isfinite(results1['best_k']) and np.isfinite(results2['best_k']):
        sv_ll_half1 = subjective_value(
            df_clean['amount'].values,
            df_clean['Delay'].values,
            results1['best_k'],
        )
        sv_ll_half2 = subjective_value(
            df_clean['amount'].values,
            df_clean['Delay'].values,
            results2['best_k'],
        )
        sv_corr = np.corrcoef(sv_ll_half1, sv_ll_half2)[0, 1]
        sv_mae = np.mean(np.abs(sv_ll_half1 - sv_ll_half2))
    else:
        sv_corr = np.nan
        sv_mae = np.nan

    return {
        'k_half1': results1['best_k'],
        'k_half2': results2['best_k'],
        'beta_half1': results1['best_beta'],
        'beta_half2': results2['best_beta'],
        'k_diff': k_diff,
        'beta_diff': beta_diff,
        'sv_correlation': sv_corr,
        'sv_mae': sv_mae,
        'half1_n_invalid_evaluations': results1['n_invalid_evaluations_total'],
        'half2_n_invalid_evaluations': results2['n_invalid_evaluations_total'],
        'half1_n_warnings': results1['n_warnings_total'],
        'half2_n_warnings': results2['n_warnings_total'],
        'half1_n_failed_initializations': results1['n_failed_initializations'],
        'half2_n_failed_initializations': results2['n_failed_initializations'],
    }


__all__ = [
    'SS_AMOUNT',
    'subjective_value',
    'compute_trialwise_measures',
    'neg_log_likelihood',
    'fit_subject_collect_all_attempts',
    'split_half_reliability',
]
