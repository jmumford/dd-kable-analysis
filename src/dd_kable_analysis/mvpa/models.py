from __future__ import annotations

"""
Modeling utilities for MVPA decoding.

Currently provides:
- nested, group-aware cross-validated ridge regression
- nested, group-aware cross-validated logistic classification
- out-of-sample predictions for every sample
- fold-safe cross-validated metrics computed from training-data baselines where relevant
"""

from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
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


def nested_groupcv_logreg_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    Cs: Iterable[float] | None = None,
    positive_label: int = 1,
    threshold: float = 0.5,
    max_iter: int = 1000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Nested group CV logistic classification with out-of-sample probabilities.

    Outer loop:
      - GroupKFold with n_splits = number of unique groups (typically leave-one-run-out)
      - used to generate out-of-sample probabilities/predictions for every sample

    Inner loop:
      - GroupKFold within the outer training set
      - selects inverse regularization strength C by minimizing log loss

    Important: X is standardized within each fit via a Pipeline(StandardScaler,
    LogisticRegression).

    Parameters
    ----------
    X
        Feature matrix of shape (n_samples, n_features).
    y
        Binary target vector of shape (n_samples,). Values must be 0/1.
    groups
        Group labels of shape (n_samples,) used for GroupKFold splits (e.g., run IDs).
        Must contain at least 3 unique groups for nested CV.
    Cs
        Candidate inverse regularization strengths to search over. If None, uses
        a default log grid.
    positive_label
        Label treated as the positive class when returning probabilities and metrics.
        Currently expects binary 0/1 coding with positive_label=1.
    threshold
        Probability threshold used to convert OOS probabilities into hard labels.
    max_iter
        Maximum iterations for LogisticRegression solver.
    verbose
        If True, prints outer fold info and best inner Cs.

    Returns
    -------
    y_prob_oos
        Out-of-sample predicted probability for the positive class, shape (n_samples,).
    y_pred_oos
        Out-of-sample predicted labels, shape (n_samples,).
    info
        Dictionary of summary metrics and fold details. Keys include:
          - n_samples, n_features, n_groups
          - log_loss, roc_auc, sensitivity, specificity, balanced_accuracy, accuracy
          - chosen_Cs: list of selected C per outer fold
          - outer_folds: list of dicts with per-fold metrics and class counts
          - c_grid
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if positive_label != 1:
        raise ValueError('Only positive_label=1 is currently supported.')

    if not np.all(np.isfinite(X)):
        raise ValueError('X contains non-finite values.')

    uniq_y = np.unique(y)
    if not np.array_equal(np.sort(uniq_y), np.array([0, 1])):
        raise ValueError(f'y must be binary with values [0, 1]; found {uniq_y}')

    y = y.astype(int)

    uniq_groups = np.unique(groups)
    n_groups = len(uniq_groups)
    if n_groups < 3:
        raise ValueError(
            f'Need >=3 groups for nested CV classification; found {n_groups}: {uniq_groups}'
        )

    if Cs is None:
        Cs = 10.0 ** np.linspace(-4, 4, 17)
    Cs = np.asarray(list(Cs), dtype=float)

    def make_pipe(C: float) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler(with_mean=True, with_std=True)),
                (
                    'logreg',
                    LogisticRegression(
                        C=float(C),
                        penalty='l2',
                        solver='liblinear',
                        max_iter=max_iter,
                    ),
                ),
            ]
        )

    outer_cv = GroupKFold(n_splits=n_groups)
    y_prob_oos = np.full(y.shape, np.nan, dtype=float)
    y_pred_oos = np.full(y.shape, -1, dtype=int)

    chosen_Cs: list[float] = []
    outer_folds: list[dict[str, Any]] = []

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

        y_tr_outer = y[tr_idx]
        if np.unique(y_tr_outer).size < 2:
            raise RuntimeError(
                f'Outer training fold {fold} has only one class: {np.unique(y_tr_outer)}'
            )

        inner_groups = groups[tr_idx]
        inner_uniq = np.unique(inner_groups)
        if len(inner_uniq) < 2:
            raise RuntimeError('Inner CV needs >=2 groups inside outer train.')

        inner_cv = GroupKFold(n_splits=len(inner_uniq))

        mean_losses: list[float] = []
        for C in Cs:
            losses: list[float] = []
            for tr2, va2 in inner_cv.split(X[tr_idx], y[tr_idx], groups=inner_groups):
                tr = tr_idx[tr2]
                va = tr_idx[va2]

                y_inner_train = y[tr]
                if np.unique(y_inner_train).size < 2:
                    losses.append(np.inf)
                    continue

                pipe = make_pipe(float(C))
                pipe.fit(X[tr], y_inner_train)
                prob = pipe.predict_proba(X[va])[:, 1]
                losses.append(float(log_loss(y[va], prob, labels=[0, 1])))

            mean_losses.append(float(np.mean(losses)))

        mean_losses_arr = np.asarray(mean_losses, dtype=float)
        best_C = float(Cs[int(np.argmin(mean_losses_arr))])
        chosen_Cs.append(best_C)

        if verbose:
            best_k = np.argsort(mean_losses_arr)[:5]
            print(f'  best C={best_C:.4g}')
            print('  top inner (C, mean log_loss):')
            for j in best_k:
                print(f'    {Cs[j]:.4g}  {mean_losses_arr[j]:.4g}')

        pipe = make_pipe(best_C)
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1]
        pred = (prob >= threshold).astype(int)

        y_prob_oos[te_idx] = prob
        y_pred_oos[te_idx] = pred

        y_te = y[te_idx]
        tp = int(np.sum((y_te == 1) & (pred == 1)))
        tn = int(np.sum((y_te == 0) & (pred == 0)))
        fp = int(np.sum((y_te == 0) & (pred == 1)))
        fn = int(np.sum((y_te == 1) & (pred == 0)))

        sens_denom = tp + fn
        spec_denom = tn + fp
        fold_sensitivity = float(tp / sens_denom) if sens_denom > 0 else np.nan
        fold_specificity = float(tn / spec_denom) if spec_denom > 0 else np.nan
        fold_bal_acc = (
            float(np.nanmean([fold_sensitivity, fold_specificity]))
            if np.isfinite(fold_sensitivity) or np.isfinite(fold_specificity)
            else np.nan
        )
        fold_auc = (
            float(roc_auc_score(y_te, prob)) if np.unique(y_te).size == 2 else np.nan
        )
        fold_log_loss = float(log_loss(y_te, prob, labels=[0, 1]))
        fold_acc = float(np.mean(pred == y_te))

        outer_folds.append(
            dict(
                fold=int(fold),
                test_groups=[str(g) for g in te_groups.tolist()],
                best_C=float(best_C),
                n_train=int(len(tr_idx)),
                n_test=int(len(te_idx)),
                n_pos_test=int(np.sum(y_te == 1)),
                n_neg_test=int(np.sum(y_te == 0)),
                sensitivity=fold_sensitivity,
                specificity=fold_specificity,
                balanced_accuracy=fold_bal_acc,
                accuracy=fold_acc,
                roc_auc=fold_auc,
                log_loss=fold_log_loss,
            )
        )

        if verbose:
            print(
                '  outer test log_loss={:.4g} auc={} sens={} spec={}'.format(
                    fold_log_loss,
                    f'{fold_auc:.4g}' if np.isfinite(fold_auc) else 'nan',
                    f'{fold_sensitivity:.4g}'
                    if np.isfinite(fold_sensitivity)
                    else 'nan',
                    f'{fold_specificity:.4g}'
                    if np.isfinite(fold_specificity)
                    else 'nan',
                )
            )

    if not np.all(np.isfinite(y_prob_oos)):
        raise RuntimeError(
            'Some samples missing OOS probabilities. Check group splits.'
        )
    if np.any(y_pred_oos < 0):
        raise RuntimeError('Some samples missing OOS predictions. Check group splits.')

    tp = int(np.sum((y == 1) & (y_pred_oos == 1)))
    tn = int(np.sum((y == 0) & (y_pred_oos == 0)))
    fp = int(np.sum((y == 0) & (y_pred_oos == 1)))
    fn = int(np.sum((y == 1) & (y_pred_oos == 0)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    balanced_accuracy = (
        float(np.nanmean([sensitivity, specificity]))
        if np.isfinite(sensitivity) or np.isfinite(specificity)
        else np.nan
    )
    accuracy = float(np.mean(y_pred_oos == y))
    overall_auc = (
        float(roc_auc_score(y, y_prob_oos)) if np.unique(y).size == 2 else np.nan
    )
    overall_log_loss = float(log_loss(y, y_prob_oos, labels=[0, 1]))

    info: dict[str, Any] = dict(
        n_samples=int(len(y)),
        n_features=int(X.shape[1]),
        n_groups=int(n_groups),
        positive_label=int(positive_label),
        threshold=float(threshold),
        log_loss=overall_log_loss,
        roc_auc=overall_auc,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        accuracy=accuracy,
        confusion_matrix=dict(tp=tp, tn=tn, fp=fp, fn=fn),
        chosen_Cs=chosen_Cs,
        outer_folds=outer_folds,
        c_grid=Cs.tolist(),
    )

    if verbose:
        print(
            '\n[overall OOS] log_loss={:.4g} auc={} sens={} spec={} bal_acc={}'.format(
                overall_log_loss,
                f'{overall_auc:.4g}' if np.isfinite(overall_auc) else 'nan',
                f'{sensitivity:.4g}' if np.isfinite(sensitivity) else 'nan',
                f'{specificity:.4g}' if np.isfinite(specificity) else 'nan',
                f'{balanced_accuracy:.4g}' if np.isfinite(balanced_accuracy) else 'nan',
            )
        )

    return y_prob_oos, y_pred_oos, info
