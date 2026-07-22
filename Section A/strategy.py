"""
Student implementation file — submit this file only.

Strategy overview
-----------------
1. Start from the 500 free initial labels.
2. Batch-mode uncertainty sampling: repeatedly train the fixed Random Forest,
   score all remaining pool candidates, and query the samples whose predicted
   P(Left) is closest to 0.5 — until the oracle budget is exhausted (with
   runtime and budget guards). Cluster-diversity batches were evaluated and
   rejected: the tiny local gain was not reproducible across scikit-learn
   versions, while plain uncertainty selection is bit-for-bit deterministic.
3. Rebalance for F1: the test metric is F1 of the minority "Left" class under
   model.predict() (0.5 vote threshold). Duplicating positive training rows
   shifts the effective threshold; the duplication ratio is chosen by repeated
   3-fold cross-validation on the labeled data only (never the test set).
4. Train the final Random Forest on the labeled set + duplicated positives.

The runtime guards time a fit, a scoring pass and an oracle call as the loop
runs, and size the remaining work from those measurements. If the grading
machine is slow, batches shrink before the CV grid does.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils import (
    call_oracle,
    get_oracle_usage,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)

# Hard limit is 60 s per seed; the reserve covers the final fit and timing noise.
_HARD_LIMIT_SEC = 60.0
_RESERVE_SEC = 5.0
_DEADLINE_SEC = _HARD_LIMIT_SEC - _RESERVE_SEC

_BATCH_SIZE = 500

# Candidate duplication ratios (extra copies of each positive row).
_DUP_RATIOS = (1, 1.5, 2.0, 2.5)
# (repeats, grid) tried in order; the first one that fits the time left is used.
_CV_LADDER = (
    (2, _DUP_RATIOS),
    (1, _DUP_RATIOS),
)
_DEFAULT_DUP = 2.0

# Fits on duplicated data cost more than a plain loop fit.
_REBALANCED_FIT_FACTOR = 1.5
# Only reached if the query loop never ran and left no timings.
_FALLBACK_FIT_SEC = 1.0


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode unlabeled rows with the framework's encoder (public API only)."""
    work = df.copy()
    work["Attrition"] = 0  # dummy label, discarded
    X, _, _ = prepare_xy(work)
    return X


def _duplicate_positives(X, y, ids, dup: float):
    """Append `dup` extra copies of the positive rows (tiled, deterministic)."""
    pos_idx = np.where(y == 1)[0]
    n_extra = int(round(dup * len(pos_idx)))
    if n_extra == 0 or len(pos_idx) == 0:
        return X, y, ids
    reps = int(np.ceil(n_extra / len(pos_idx)))
    idx = np.tile(pos_idx, reps)[:n_extra]
    Xa = pd.concat([X, X.iloc[idx]], ignore_index=True)
    ya = np.concatenate([y, y[idx]])
    ida = np.concatenate([ids, ids[idx]])
    return Xa, ya, ida


def _fit(X, y, ids, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate-ID warning is expected
        return train_model(X, y, ids, seed=seed)


def _cv_ratio(X, y, ids, seed: int, repeats: int, grid) -> float:
    """Choose the duplication ratio by repeated 3-fold CV on labeled data only."""
    scores = {r: [] for r in grid}
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 100 * rep)
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, ytr, idtr = X.iloc[tr_idx], y[tr_idx], ids[tr_idx]
            Xva, yva = X.iloc[va_idx], y[va_idx]
            for r in grid:
                Xa, ya, ida = _duplicate_positives(Xtr, ytr, idtr, r)
                model = _fit(Xa, ya, ida, seed)
                scores[r].append(f1_score(yva, model.predict(Xva), pos_label=1))
    return max(grid, key=lambda r: float(np.mean(scores[r])))


def _pick_dup_ratio(X, y, ids, seed: int, avail: float, t_fit: float) -> float:
    """Run the largest CV configuration that fits in `avail` seconds."""
    for repeats, grid in _CV_LADDER:
        if repeats * 3 * len(grid) * t_fit <= avail:
            return _cv_ratio(X, y, ids, seed, repeats, grid)
    return _DEFAULT_DUP


def _plan_batch(left: int, avail: float, t_fit: float, t_score: float, t_per_id: float) -> int:
    """
    Largest batch size that keeps the rest of the run inside `avail` seconds.

    Falls back to a single bulk query, which pays the fit and scoring cost once
    instead of once per round, then to a partial query.
    """
    per_round = t_fit + t_score
    rounds_left = int(np.ceil(left / _BATCH_SIZE))
    if rounds_left * per_round + left * t_per_id <= avail:
        return min(_BATCH_SIZE, left)
    if per_round + left * t_per_id <= avail:
        return left
    if t_per_id <= 0.0:
        return 0
    return max(0, int((avail - per_round) / t_per_id))


def run_active_learning(seed: int):
    """
    Run active learning for the given seed and return a trained RandomForestClassifier.

    Parameters
    ----------
    seed : int
        One of {1, 2, 3}. Controls randomness and selects the initial labeled set.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained model to be evaluated on the hidden test set.
    """
    t_start = pd.Timestamp.now()

    def elapsed() -> float:
        return (pd.Timestamp.now() - t_start).total_seconds()

    labeled = load_initial_labeled(seed)
    pool = load_pool()

    have = set(labeled["Employee ID"].astype(str))
    cand = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
    cand_ids = cand["Employee ID"].astype(str).to_numpy()
    X_cand = _encode(cand)

    budget = int(get_oracle_usage()["remaining"])
    budget = min(budget, len(cand_ids))

    # Timings collected during the loop. Fits and scoring grow with the labeled
    # set, so keep the worst case; oracle cost is flat per ID, so keep the mean.
    t_fit = 0.0
    t_score = 0.0
    t_oracle_total = 0.0
    n_queried = 0

    spent = 0
    while spent < budget and len(cand_ids) > 0:
        left = min(budget - spent, len(cand_ids))

        if n_queried == 0:
            k = min(_BATCH_SIZE, left)  # first round: nothing measured yet
        else:
            tail = _REBALANCED_FIT_FACTOR * t_fit  # room for the final fit
            k = _plan_batch(
                left,
                _DEADLINE_SEC - elapsed() - tail,
                t_fit,
                t_score,
                t_oracle_total / n_queried,
            )
            if k < 1:
                break

        t0 = pd.Timestamp.now()
        X, y, ids = prepare_xy(labeled)
        model = _fit(X, y, ids, seed)
        t_fit = max(t_fit, (pd.Timestamp.now() - t0).total_seconds())

        t0 = pd.Timestamp.now()
        proba_all = model.predict_proba(X_cand)
        classes = list(model.classes_)
        if 1 in classes:  # guard against a degenerate single-class fit
            proba = proba_all[:, classes.index(1)]
        else:
            proba = np.zeros(len(X_cand))
        margin = np.abs(proba - 0.5)
        take_idx = np.argsort(margin, kind="stable")[:k]
        t_score = max(t_score, (pd.Timestamp.now() - t0).total_seconds())

        t0 = pd.Timestamp.now()
        new_rows = call_oracle(list(cand_ids[take_idx]))
        t_oracle_total += (pd.Timestamp.now() - t0).total_seconds()
        n_queried += len(take_idx)

        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        spent += len(take_idx)

        keep = np.setdiff1d(np.arange(len(cand_ids)), take_idx)
        cand_ids = cand_ids[keep]
        X_cand = X_cand.iloc[keep].reset_index(drop=True)

    # Final rebalanced model; the CV size depends on the time left.
    X, y, ids = prepare_xy(labeled)
    t_reb_fit = _REBALANCED_FIT_FACTOR * (t_fit if t_fit > 0.0 else _FALLBACK_FIT_SEC)
    dup = _pick_dup_ratio(X, y, ids, seed, _DEADLINE_SEC - elapsed() - t_reb_fit, t_reb_fit)
    Xa, ya, ida = _duplicate_positives(X, y, ids, dup)
    return _fit(Xa, ya, ida, seed)
