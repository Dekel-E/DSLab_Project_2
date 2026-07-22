
"""
Hybrid active-learning strategy.

Main ideas:
1. Keep a small random oracle sample as a representative calibration set.
2. Spend the rest of the budget in only three adaptive batches.
3. In each batch combine uncertainty, likely-positive, and random sampling.
4. Select a moderate positive-duplication ratio on the calibration set.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils import (
    call_oracle,
    get_oracle_usage,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)


ID_COLUMN = "Employee ID"
TARGET_COLUMN = "Attrition"

# 750 representative labels for choosing the final class balance.
_CALIBRATION_SIZE = 750

# Three adaptive rounds instead of ten small rounds.
_STAGE_SIZE = 1500
_QUERY_TIME_GUARD_SEC = 30.0
_RATIO_TIME_GUARD_SEC = 46.0

# Extra positive copies per original positive row.
_DUP_RATIOS = (0.5, 1.0, 1.5, 2.0, 2.5)
_DEFAULT_DUP_RATIO = 1.5


def _elapsed_seconds(start: pd.Timestamp) -> float:
    return (pd.Timestamp.now() - start).total_seconds()


def _encode_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work[TARGET_COLUMN] = 0
    X, _, _ = prepare_xy(work)
    return X


def _fit(X, y, ids, seed: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return train_model(X, y, ids, seed=seed)


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def _mixed_selection_indices(
    probabilities: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select 65% uncertain, 15% likely-positive, and 20% random rows."""
    n = len(probabilities)
    count = min(max(int(count), 0), n)
    if count == 0:
        return np.empty(0, dtype=int)

    uncertain_count = min(int(round(0.65 * count)), count)
    positive_count = min(int(round(0.15 * count)), count - uncertain_count)

    chosen = np.zeros(n, dtype=bool)
    parts: list[np.ndarray] = []

    uncertainty_order = np.argsort(np.abs(probabilities - 0.5), kind="stable")
    uncertain_idx = uncertainty_order[:uncertain_count]
    chosen[uncertain_idx] = True
    parts.append(uncertain_idx)

    if positive_count > 0:
        positive_order = np.argsort(-probabilities, kind="stable")
        positive_order = positive_order[~chosen[positive_order]]
        positive_idx = positive_order[:positive_count]
        chosen[positive_idx] = True
        parts.append(positive_idx)

    remaining_count = count - int(chosen.sum())
    if remaining_count > 0:
        remaining_idx = np.flatnonzero(~chosen)
        rng = np.random.default_rng(seed)
        random_idx = rng.choice(remaining_idx, size=remaining_count, replace=False)
        parts.append(random_idx)

    return np.concatenate(parts)


def _duplicate_positives(X, y, ids, ratio: float):
    positive_idx = np.flatnonzero(y == 1)
    extra_count = int(round(float(ratio) * len(positive_idx)))

    if len(positive_idx) == 0 or extra_count <= 0:
        return X, y, ids

    repeats = int(np.ceil(extra_count / len(positive_idx)))
    extra_idx = np.tile(positive_idx, repeats)[:extra_count]

    X_augmented = pd.concat([X, X.iloc[extra_idx]], ignore_index=True)
    y_augmented = np.concatenate([y, y[extra_idx]])
    ids_augmented = np.concatenate([ids, ids[extra_idx]])
    return X_augmented, y_augmented, ids_augmented


def _choose_duplication_ratio(
    training_rows: pd.DataFrame,
    calibration_rows: pd.DataFrame,
    seed: int,
    start: pd.Timestamp,
) -> float:
    """Use a representative random holdout and only a few additional fits."""
    if calibration_rows.empty:
        return _DEFAULT_DUP_RATIO

    X_train, y_train, train_ids = prepare_xy(training_rows)
    X_cal, y_cal, _ = prepare_xy(calibration_rows)

    best_ratio = _DEFAULT_DUP_RATIO
    best_score = -1.0

    for ratio in _DUP_RATIOS:
        if _elapsed_seconds(start) > _RATIO_TIME_GUARD_SEC:
            break

        Xa, ya, ida = _duplicate_positives(X_train, y_train, train_ids, ratio)
        model = _fit(Xa, ya, ida, seed)
        score = f1_score(y_cal, model.predict(X_cal), pos_label=1, zero_division=0)

        # Prefer the smaller ratio when scores are equal.
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and ratio < best_ratio
        ):
            best_score = float(score)
            best_ratio = ratio

    return best_ratio


def run_active_learning(seed: int):
    start = pd.Timestamp.now()

    initial = load_initial_labeled(seed)
    pool = load_pool().copy()

    initial_ids = set(initial[ID_COLUMN].astype(str))
    pool["_id_str"] = pool[ID_COLUMN].astype(str)
    candidates = (
        pool[~pool["_id_str"].isin(initial_ids)]
        .drop_duplicates("_id_str")
        .sort_values("_id_str", kind="mergesort")
        .reset_index(drop=True)
    )

    budget = min(int(get_oracle_usage()["remaining"]), len(candidates))
    if budget <= 0:
        X, y, ids = prepare_xy(initial)
        return _fit(X, y, ids, seed)

    # Random labels are kept out of active-model fitting so they remain a clean,
    # representative set for choosing the final duplication ratio.
    calibration_size = min(_CALIBRATION_SIZE, budget)
    rng = np.random.default_rng(seed + 10_000)
    calibration_idx = rng.choice(
        len(candidates), size=calibration_size, replace=False
    )
    calibration_rows = call_oracle(
        candidates.iloc[calibration_idx]["_id_str"].tolist()
    )

    keep = np.ones(len(candidates), dtype=bool)
    keep[calibration_idx] = False
    candidates = candidates.iloc[np.flatnonzero(keep)].reset_index(drop=True)

    candidate_ids = candidates["_id_str"].to_numpy()
    X_candidates = _encode_unlabeled(candidates.drop(columns=["_id_str"]))

    active_labeled = initial.copy()
    active_budget = min(budget - calibration_size, len(candidate_ids))
    spent = 0
    round_index = 0

    while spent < active_budget and len(candidate_ids) > 0:
        X_train, y_train, train_ids = prepare_xy(active_labeled)
        model = _fit(X_train, y_train, train_ids, seed)

        remaining_budget = active_budget - spent
        if _elapsed_seconds(start) > _QUERY_TIME_GUARD_SEC:
            batch_size = remaining_budget
        else:
            batch_size = min(_STAGE_SIZE, remaining_budget)

        probabilities = _positive_probability(model, X_candidates)
        selected_idx = _mixed_selection_indices(
            probabilities,
            batch_size,
            seed=seed * 100 + round_index,
        )

        queried = call_oracle(candidate_ids[selected_idx].tolist())
        active_labeled = pd.concat([active_labeled, queried], ignore_index=True)
        spent += len(selected_idx)
        round_index += 1

        keep = np.ones(len(candidate_ids), dtype=bool)
        keep[selected_idx] = False
        keep_idx = np.flatnonzero(keep)
        candidate_ids = candidate_ids[keep_idx]
        X_candidates = X_candidates.iloc[keep_idx].reset_index(drop=True)

    duplication_ratio = _choose_duplication_ratio(
        active_labeled,
        calibration_rows,
        seed,
        start,
    )

    all_labeled = pd.concat([active_labeled, calibration_rows], ignore_index=True)
    X_final, y_final, final_ids = prepare_xy(all_labeled)
    Xa, ya, ida = _duplicate_positives(
        X_final, y_final, final_ids, duplication_ratio
    )
    return _fit(Xa, ya, ida, seed)
