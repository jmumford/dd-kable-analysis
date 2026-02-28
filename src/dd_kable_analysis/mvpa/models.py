from __future__ import annotations

"""
Modeling utilities for MVPA decoding.

Currently provides:
- nested, group-aware cross-validated ridge regression
- out-of-sample predictions for every sample
- fold-safe cross-validated R^2 (baseline mean computed from training data per fold)
"""

from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def nested_groupcv_ridge_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Nested group CV ridge regression with out-of-sample predictions.

    Outer loop:
      - GroupKFold with n_splits = number of unique groups (typically leave-one-run-out)
      - used to generate out-of-sample predictions for every sample

    Inner loop:
      - GroupKFold within the outer training set
      - selects ridge alpha by minimizing mean squared error (MSE)

    Important: X is standardized within each fit via a Pipeline(StandardScaler, Ridge).

    Parameters
    ----------
    X
        Feature matrix of shape (n_samples, n_features).
    y
        Target vector of shape (n_samples,). Must be numeric.
    groups
        Group labels of shape (n_samples,) used for GroupKFold splits (e.g., run IDs).
        Must contain at least 3 unique groups for nested CV.
    alphas
        Candidate ridge penalties to search over. If None, uses a default log grid.
    verbose
        If True, prints outer fold info and best inner alphas.

    Returns
    -------
    y_pred_oos
        Out-of-sample predictions for each sample, shape (n_samples,).
        Every entry is predicted from a model that did not train on its group.
    info
        Dictionary of summary metrics and fold details. Keys include:
          - n_samples, n_features, n_groups
          - rmse, mse
          - r: corr(y, y_pred_oos) across all samples
          - r2: R^2 using global mean baseline (less strict; kept for reference)
          - r2_cv: fold-safe cross-validated R^2 using training-mean baseline per outer fold
          - chosen_alphas: list of selected alpha per outer fold
          - outer_folds: list of dicts with per-fold metrics (mse, r, sse, baseline sse, etc.)

    Notes
    -----
    Fold-safe R^2_cv is computed as:

      For each outer fold k:
        SSE_k = sum_{i in test_k} (y_i - yhat_i)^2
        ybar_train_k = mean(y_train_k)
        SSE_base_k = sum_{i in test_k} (y_i - ybar_train_k)^2

      Aggregate across folds:
        R^2_cv = 1 - (sum_k SSE_k) / (sum_k SSE_base_k)

    Negative R^2_cv is possible and indicates worse-than-baseline predictions.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    n_groups = len(uniq)
    if n_groups < 3:
        raise ValueError(f'Need >=3 groups for nested CV; found {n_groups}: {uniq}')

    if alphas is None:
        alphas = 10.0 ** np.linspace(-2, 6, 20)
    alphas = np.asarray(alphas, dtype=float)

    def make_pipe(alpha: float) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler(with_mean=True, with_std=True)),
                ('ridge', Ridge(alpha=alpha, fit_intercept=True)),
            ]
        )

    outer_cv = GroupKFold(n_splits=n_groups)
    y_pred_oos = np.full(y.shape, np.nan, dtype=float)

    chosen_alphas: list[float] = []
    outer_folds: list[dict[str, Any]] = []

    # accumulators for fold-safe R2_cv
    sse_total = 0.0
    sse_base_total = 0.0

    for fold, (tr_idx, te_idx) in enumerate(
        outer_cv.split(X, y, groups=groups), start=1
    ):
        te_groups = np.unique(groups[te_idx])
        tr_groups = np.unique(groups[tr_idx])

        if verbose:
            print(
                f'\n[outer {fold}/{n_groups}] test groups={te_groups} train groups={tr_groups}'
            )
            print(f'  n_train={len(tr_idx)} n_test={len(te_idx)}')

        inner_groups = groups[tr_idx]
        inner_uniq = np.unique(inner_groups)
        if len(inner_uniq) < 2:
            raise RuntimeError('Inner CV needs >=2 groups inside outer train.')

        inner_cv = GroupKFold(n_splits=len(inner_uniq))

        mean_mses: list[float] = []
        for a in alphas:
            mses: list[float] = []
            for tr2, va2 in inner_cv.split(X[tr_idx], y[tr_idx], groups=inner_groups):
                tr = tr_idx[tr2]
                va = tr_idx[va2]
                pipe = make_pipe(float(a))
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[va])
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            mean_mses.append(float(np.mean(mses)))

        mean_mses_arr = np.asarray(mean_mses, dtype=float)
        best_alpha = float(alphas[int(np.argmin(mean_mses_arr))])
        chosen_alphas.append(best_alpha)

        if verbose:
            best_k = np.argsort(mean_mses_arr)[:5]
            print(f'  best alpha={best_alpha:.4g}')
            print('  top inner (alpha, mean MSE):')
            for j in best_k:
                print(f'    {alphas[j]:.4g}  {mean_mses_arr[j]:.4g}')

        pipe = make_pipe(best_alpha)
        pipe.fit(X[tr_idx], y[tr_idx])
        yhat = pipe.predict(X[te_idx])
        y_pred_oos[te_idx] = yhat

        y_te = y[te_idx]
        y_tr = y[tr_idx]

        fold_mse = float(np.mean((y_te - yhat) ** 2))
        fold_r = float(np.corrcoef(y_te, yhat)[0, 1]) if len(te_idx) > 2 else np.nan

        # fold-safe SSE and baseline SSE (training mean baseline)
        sse_k = float(np.sum((y_te - yhat) ** 2))
        ybar_tr = float(np.mean(y_tr))
        sse_base_k = float(np.sum((y_te - ybar_tr) ** 2))
        sse_total += sse_k
        sse_base_total += sse_base_k

        outer_folds.append(
            dict(
                fold=int(fold),
                test_groups=[str(g) for g in te_groups.tolist()],
                best_alpha=float(best_alpha),
                test_mse=float(fold_mse),
                test_r=float(fold_r) if np.isfinite(fold_r) else np.nan,
                sse_k=float(sse_k),
                sse_base_k=float(sse_base_k),
                ybar_train=float(ybar_tr),
                n_train=int(len(tr_idx)),
                n_test=int(len(te_idx)),
            )
        )

        if verbose:
            print(f'  outer test mse={fold_mse:.4g} r={fold_r:.4g}')

    ok = np.isfinite(y_pred_oos)
    if not np.all(ok):
        raise RuntimeError('Some samples missing OOS predictions. Check group splits.')

    mse = float(np.mean((y - y_pred_oos) ** 2))
    rmse = float(np.sqrt(mse))

    # correlation is fine (all preds are OOS)
    r = float(np.corrcoef(y, y_pred_oos)[0, 1])

    # r2 (global-mean baseline)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - y_pred_oos) ** 2))
    r2_globalmean = float(1.0 - sse / sst) if sst > 0 else np.nan

    # fold-safe R2_cv
    r2_cv = float(1.0 - sse_total / sse_base_total) if sse_base_total > 0 else np.nan

    info: dict[str, Any] = dict(
        n_samples=int(len(y)),
        n_features=int(X.shape[1]),
        n_groups=int(n_groups),
        rmse=float(rmse),
        mse=float(mse),
        r=float(r),
        r2_cv=float(r2_cv),
        r2=float(r2_globalmean),
        chosen_alphas=chosen_alphas,
        outer_folds=outer_folds,
        alpha_grid=alphas.tolist(),
    )

    if verbose:
        print(
            '\n[overall OOS] rmse={:.4g} mse={:.4g} r={:.4g} r2_cv={:.4g} (r2_globalmean={:.4g})'.format(
                rmse, mse, r, r2_cv, r2_globalmean
            )
        )

    return y_pred_oos, info
