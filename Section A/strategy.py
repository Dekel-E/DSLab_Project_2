"""
Student implementation file — submit this file only.

Strategy overview
-----------------
1. Start from the 500 free initial labels.
2. Batch-mode uncertainty sampling: repeatedly train the fixed Random Forest,
   score all remaining pool candidates, and query the samples whose predicted
   P(Left) is closest to 0.5 — until the oracle budget is exhausted. Cluster-
   diversity batches were evaluated and rejected: the tiny local gain was not
   reproducible across scikit-learn versions, while plain uncertainty selection
   is bit-for-bit deterministic.
3. Rebalance for F1: the test metric is F1 of the minority "Left" class under
   model.predict() (0.5 vote threshold). Duplicating positive training rows
   shifts the effective threshold; the duplication ratio is chosen by repeated
   3-fold cross-validation on the labeled data only (never the test set).
4. Train the final Random Forest on the labeled set + duplicated positives.

Runtime safety
--------------
The 60 s/seed limit is enforced on the grading machine, which may be markedly
slower than the development machine. Instead of fixed wall-clock thresholds
(which cannot know how long the work they gate will take), this file *measures*
the cost of its own building blocks as it runs — one Random Forest fit, one
pool-scoring pass, one oracle label — and plans every remaining step against
the deadline.

On a fast machine no cap ever binds, so behaviour is identical to the unguarded
strategy. On a slow machine the plan degrades in decreasing order of value:
full-size query batches -> a single bulk query -> fewer queries (the learning
curve plateaus near ~2,500 labels, so this is the cheapest thing to give up);
full CV grid -> reduced CV grid -> the fixed fallback ratio.

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

# Deadline. The grader times run_active_learning() alone; the reserve absorbs
# interpreter jitter, GC pauses and the cost model's own estimation error.
_HARD_LIMIT_SEC = 60.0
_RESERVE_SEC = 5.0
_DEADLINE_SEC = _HARD_LIMIT_SEC - _RESERVE_SEC

_BATCH_SIZE = 500

# Candidate duplication ratios (extra copies of each positive row).
_DUP_RATIOS = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
# Rungs tried in order; the first one that fits in the remaining time is used.
# Measured cost on the dev machine: 36 / 18 / 9 fits ~= 4.8 / 2.8 / 1.1 s.
_CV_LADDER = (
    (2, _DUP_RATIOS),
    (1, _DUP_RATIOS),
    (1, (1.5, 2.0, 2.5)),
)
_DEFAULT_DUP = 2.0

# A CV or final fit trains on duplicated positives and also predicts, so it
# costs more than a plain loop fit. Measured ratio ~1.3; rounded up for safety.
_REBALANCED_FIT_FACTOR = 1.5
# Only used if the query loop never ran, leaving no measurements to plan with.
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
    """Run the richest CV rung that fits in `avail` seconds; else the fallback."""
    for repeats, grid in _CV_LADDER:
        if repeats * 3 * len(grid) * t_fit <= avail:
            return _cv_ratio(X, y, ids, seed, repeats, grid)
    return _DEFAULT_DUP


def _plan_batch(left: int, avail: float, t_fit: float, t_score: float, t_per_id: float) -> int:
    """
    Largest batch size that keeps the remaining plan inside `avail` seconds.

    Prefers the standard batch; falls back to one bulk query (which pays the
    per-round fit and scoring cost once instead of once per round), then to a
    partial query. Returns 0 when not even one more label is affordable.
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

    # Cost model, measured on this machine as the loop runs. Fits and scoring
    # use the worst case seen (they grow with the labeled set); oracle cost is
    # per-ID and flat, so its mean is the right estimator.
    t_fit = 0.0
    t_score = 0.0
    t_oracle_total = 0.0
    n_queried = 0

    spent = 0
    while spent < budget and len(cand_ids) > 0:
        left = min(budget - spent, len(cand_ids))

        if n_queried == 0:
            # No measurements yet. One standard batch is safe on any plausible
            # machine, and it is what calibrates the cost model.
            k = min(_BATCH_SIZE, left)
        else:
            tail = _REBALANCED_FIT_FACTOR * t_fit  # the final fit still has to happen
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

    # Final rebalanced model, with the CV grid sized to the time that is left.
    X, y, ids = prepare_xy(labeled)
    t_reb_fit = _REBALANCED_FIT_FACTOR * (t_fit if t_fit > 0.0 else _FALLBACK_FIT_SEC)
    dup = _pick_dup_ratio(X, y, ids, seed, _DEADLINE_SEC - elapsed() - t_reb_fit, t_reb_fit)
    Xa, ya, ida = _duplicate_positives(X, y, ids, dup)
    return _fit(Xa, ya, ida, seed)
