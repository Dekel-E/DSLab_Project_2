"""
Student implementation file — submit this file only.

Strategy overview
-----------------
1. Start from the 500 free initial labels.
2. Batch-mode uncertainty sampling with diversity: repeatedly train the fixed
   Random Forest, score all remaining pool candidates, take the most uncertain
   region (predicted P(Left) closest to 0.5), cluster it, and query the most
   uncertain sample per cluster — until the oracle budget is exhausted
   (with runtime and budget guards). Diverse batches avoid spending budget on
   near-duplicate borderline samples.
3. Rebalance for F1: the test metric is F1 of the minority "Left" class under
   model.predict() (0.5 vote threshold). Duplicating positive training rows
   shifts the effective threshold; the duplication ratio is chosen by repeated
   3-fold cross-validation on the labeled data only (never the test set).
4. Train the final Random Forest on the labeled set + duplicated positives.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
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

# Runtime guards (hard limit is 60 s per seed on the grading machine):
# stop querying at _QUERY_TIME_LIMIT to leave room for the final fit, and
# skip the rebalancing CV at _CV_TIME_LIMIT, falling back to _DEFAULT_DUP.
_QUERY_TIME_LIMIT_SEC = 30.0
_CV_TIME_LIMIT_SEC = 42.0
_BATCH_SIZE = 500
# Candidate duplication ratios (extra copies of each positive row).
_DUP_RATIOS = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
_DEFAULT_DUP = 2.0
_CV_REPEATS = 2


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


def _select_diverse_batch(margin: np.ndarray, X_cand: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Pick k candidates: cluster the most uncertain region, one per cluster."""
    if k >= len(margin):
        return np.arange(len(margin))
    top = np.argsort(margin, kind="stable")[: min(4 * k, len(margin))]
    if k < 2 or len(top) <= k:
        return top[:k]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3)
    clusters = km.fit_predict(X_cand.iloc[top])
    take = []
    for c in range(k):
        members = top[clusters == c]
        if len(members):
            take.append(members[np.argmin(margin[members])])
    take = np.array(take, dtype=int)
    if len(take) < k:  # backfill from the most uncertain not yet taken
        taken = set(take.tolist())
        extra = [i for i in top if i not in taken][: k - len(take)]
        take = np.concatenate([take, np.array(extra, dtype=int)])
    return take


def _pick_dup_ratio(X, y, ids, seed: int) -> float:
    """Choose the duplication ratio by repeated 3-fold CV on labeled data only."""
    scores = {r: [] for r in _DUP_RATIOS}
    for rep in range(_CV_REPEATS):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 100 * rep)
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, ytr, idtr = X.iloc[tr_idx], y[tr_idx], ids[tr_idx]
            Xva, yva = X.iloc[va_idx], y[va_idx]
            for r in _DUP_RATIOS:
                Xa, ya, ida = _duplicate_positives(Xtr, ytr, idtr, r)
                model = _fit(Xa, ya, ida, seed)
                scores[r].append(f1_score(yva, model.predict(Xva), pos_label=1))
    return max(_DUP_RATIOS, key=lambda r: float(np.mean(scores[r])))


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

    labeled = load_initial_labeled(seed)
    pool = load_pool()

    have = set(labeled["Employee ID"].astype(str))
    cand = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
    cand_ids = cand["Employee ID"].astype(str).to_numpy()
    X_cand = _encode(cand)

    budget = int(get_oracle_usage()["remaining"])
    budget = min(budget, len(cand_ids))

    spent = 0
    while spent < budget and len(cand_ids) > 0:
        elapsed = (pd.Timestamp.now() - t_start).total_seconds()
        if elapsed > _QUERY_TIME_LIMIT_SEC:
            k = budget - spent  # out of time: spend the rest in one batch
        else:
            k = min(_BATCH_SIZE, budget - spent)
        k = min(k, len(cand_ids))

        X, y, ids = prepare_xy(labeled)
        model = _fit(X, y, ids, seed)

        proba_all = model.predict_proba(X_cand)
        classes = list(model.classes_)
        if 1 in classes:  # guard against a degenerate single-class fit
            proba = proba_all[:, classes.index(1)]
        else:
            proba = np.zeros(len(X_cand))
        margin = np.abs(proba - 0.5)
        take_idx = _select_diverse_batch(margin, X_cand, k, seed)

        new_rows = call_oracle(list(cand_ids[take_idx]))
        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        spent += len(take_idx)

        keep = np.setdiff1d(np.arange(len(cand_ids)), take_idx)
        cand_ids = cand_ids[keep]
        X_cand = X_cand.iloc[keep].reset_index(drop=True)

    # Final rebalanced model. Skip the CV if time is short (slow grader).
    X, y, ids = prepare_xy(labeled)
    elapsed = (pd.Timestamp.now() - t_start).total_seconds()
    if elapsed > _CV_TIME_LIMIT_SEC:
        dup = _DEFAULT_DUP
    else:
        dup = _pick_dup_ratio(X, y, ids, seed)
    Xa, ya, ida = _duplicate_positives(X, y, ids, dup)
    return _fit(Xa, ya, ida, seed)
