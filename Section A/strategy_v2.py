"""
Student implementation file — submit this file only.

Implement run_active_learning(seed) to run your active learning strategy and return
a trained RandomForestClassifier. You may add helper functions in this file only.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

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


def _remaining_oracle_budget(default: int = 5000) -> int:
    """Return the remaining unique oracle-query budget."""
    try:
        usage = get_oracle_usage()
        remaining = usage.get("remaining", default)
        return max(int(remaining), 0)
    except Exception:
        return default


def _query_random_unique(
    pool: pd.DataFrame,
    labeled_ids: set[str],
    seed: int,
    budget: int | None = None,
) -> pd.DataFrame:
    """Query a deterministic random subset of unique unlabeled Employee IDs."""
    candidates = pool.copy()
    candidates["_employee_id_str"] = candidates[ID_COLUMN].astype(str)

    candidates = candidates[~candidates["_employee_id_str"].isin(labeled_ids)]
    candidates = candidates.drop_duplicates("_employee_id_str")
    candidates = candidates.sort_values("_employee_id_str", kind="mergesort").reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame(columns=list(pool.columns) + [TARGET_COLUMN])

    if budget is None:
        target_size = min(_remaining_oracle_budget(), len(candidates))
    else:
        target_size = min(max(int(budget), 0), len(candidates))
    if target_size == 0:
        return pd.DataFrame(columns=list(pool.columns) + [TARGET_COLUMN])

    sampled = candidates.sample(n=target_size, random_state=seed)
    query_ids = sampled["_employee_id_str"].tolist()

    return call_oracle(query_ids)


def _score_pool_positive_probability(
    model,
    candidates: pd.DataFrame,
) -> np.ndarray:
    """Return model probabilities for the positive Attrition class."""
    scoring_df = candidates.drop(columns=["_employee_id_str"], errors="ignore").copy()
    scoring_df[TARGET_COLUMN] = 0
    X_pool, _, _ = prepare_xy(scoring_df)

    probabilities = model.predict_proba(X_pool)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(candidates), dtype=float)

    return probabilities[:, classes.index(1)]


def _query_random_with_uncertainty(
    model,
    pool: pd.DataFrame,
    labeled_ids: set[str],
    seed: int,
    budget: int = 5000,
    uncertain_budget: int = 1000,
) -> pd.DataFrame:
    """Query uncertain samples first, then fill the rest randomly."""
    candidates = pool.copy()
    candidates["_employee_id_str"] = candidates[ID_COLUMN].astype(str)
    candidates = candidates[~candidates["_employee_id_str"].isin(labeled_ids)]
    candidates = candidates.drop_duplicates("_employee_id_str")
    candidates = candidates.sort_values("_employee_id_str", kind="mergesort").reset_index(drop=True)

    target_size = min(len(candidates), max(int(budget), 0))
    if target_size == 0:
        return pd.DataFrame(columns=list(pool.columns) + [TARGET_COLUMN])

    if model is None or uncertain_budget <= 0:
        sampled = candidates.sample(n=target_size, random_state=seed)
        return call_oracle(sampled["_employee_id_str"].tolist())

    candidates = candidates.copy()
    positive_probability = _score_pool_positive_probability(model, candidates)
    candidates["_uncertainty"] = np.abs(positive_probability - 0.5)

    selected_parts = []
    selected_ids: set[str] = set()

    uncertainty_count = min(int(uncertain_budget), target_size)
    if uncertainty_count > 0:
        uncertain = candidates.sort_values(
            ["_uncertainty", "_employee_id_str"],
            kind="mergesort",
        ).head(uncertainty_count)
        selected_parts.append(uncertain)
        selected_ids.update(uncertain["_employee_id_str"])

    remaining = target_size - len(selected_ids)
    if remaining > 0:
        random_fill = candidates[
            ~candidates["_employee_id_str"].isin(selected_ids)
        ].sample(n=remaining, random_state=seed)
        selected_parts.append(random_fill)

    selected = pd.concat(selected_parts, ignore_index=True)
    return call_oracle(selected["_employee_id_str"].tolist())


def _query_positive_uncertain_random(
    model,
    pool: pd.DataFrame,
    labeled_ids: set[str],
    seed: int,
    budget: int = 5000,
    positive_budget: int = 1500,
    uncertain_budget: int = 1000,
) -> pd.DataFrame:
    """Query a high-risk/uncertain/random mix using only oracle labels."""
    candidates = pool.copy()
    candidates["_employee_id_str"] = candidates[ID_COLUMN].astype(str)
    candidates = candidates[~candidates["_employee_id_str"].isin(labeled_ids)]
    candidates = candidates.drop_duplicates("_employee_id_str")
    candidates = candidates.sort_values("_employee_id_str", kind="mergesort").reset_index(drop=True)

    target_size = min(len(candidates), max(int(budget), 0))
    if target_size == 0:
        return pd.DataFrame(columns=list(pool.columns) + [TARGET_COLUMN])

    if model is None:
        return _query_random_unique(pool, labeled_ids, seed, budget=target_size)

    candidates = candidates.copy()
    positive_probability = _score_pool_positive_probability(model, candidates)
    candidates["_positive_probability"] = positive_probability
    candidates["_uncertainty"] = np.abs(positive_probability - 0.5)

    selected_parts = []
    selected_ids: set[str] = set()

    def add_selection(frame: pd.DataFrame, count: int) -> None:
        nonlocal selected_parts, selected_ids
        remaining_slots = target_size - len(selected_ids)
        if count <= 0 or remaining_slots <= 0:
            return

        selected = frame[
            ~frame["_employee_id_str"].isin(selected_ids)
        ].head(min(int(count), remaining_slots))
        if not selected.empty:
            selected_parts.append(selected)
            selected_ids.update(selected["_employee_id_str"])

    add_selection(
        candidates.sort_values(
            ["_positive_probability", "_employee_id_str"],
            ascending=[False, True],
            kind="mergesort",
        ),
        positive_budget,
    )
    add_selection(
        candidates.sort_values(
            ["_uncertainty", "_employee_id_str"],
            kind="mergesort",
        ),
        uncertain_budget,
    )

    remaining = target_size - len(selected_ids)
    if remaining > 0:
        random_fill = candidates[
            ~candidates["_employee_id_str"].isin(selected_ids)
        ].sample(n=remaining, random_state=seed)
        selected_parts.append(random_fill)

    selected = pd.concat(selected_parts, ignore_index=True)
    return call_oracle(selected["_employee_id_str"].tolist())


def _oversample_positive(
    X: pd.DataFrame,
    y: np.ndarray,
    employee_ids: np.ndarray,
    seed: int = 0,
    target_positive_rate: float = 0.75,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Duplicate positive rows as evenly as possible with seeded tie-breaking."""
    positive_idx = np.flatnonzero(y == 1)
    negative_count = int((y == 0).sum())
    positive_count = len(positive_idx)

    if positive_count == 0 or positive_count / len(y) >= target_positive_rate:
        return X, y, employee_ids

    desired_positive_count = int(
        target_positive_rate * negative_count / max(1.0 - target_positive_rate, 1e-9)
    )

    extra_count = max(0, desired_positive_count - positive_count)
    if extra_count == 0:
        return X, y, employee_ids

    full_repeats, remainder = divmod(extra_count, positive_count)

    extra_parts = []

    if full_repeats > 0:
        extra_parts.append(np.tile(positive_idx, full_repeats))

    if remainder > 0:
        rng = np.random.default_rng(seed)
        remainder_idx = rng.choice(
            positive_idx,
            size=remainder,
            replace=False,
        )
        extra_parts.append(remainder_idx)

    extra_idx = np.concatenate(extra_parts)

    X_extra = X.iloc[extra_idx]
    y_extra = y[extra_idx]
    ids_extra = employee_ids[extra_idx]

    X_balanced = pd.concat([X, X_extra], ignore_index=True)
    y_balanced = np.concatenate([y, y_extra])
    ids_balanced = np.concatenate([employee_ids, ids_extra])

    return X_balanced, y_balanced, ids_balanced


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
    labeled = load_initial_labeled(seed)
    pool = load_pool()

    labeled_ids = set(labeled[ID_COLUMN].astype(str))
    X_initial, y_initial, initial_ids = prepare_xy(labeled)
    initial_model = train_model(
        X_initial,
        y_initial,
        initial_ids,
        seed=seed,
    )

    queried = _query_positive_uncertain_random(
        model=initial_model,
        pool=pool,
        labeled_ids=labeled_ids,
        seed=seed,
    )

    if not queried.empty:
        labeled = pd.concat([labeled, queried], ignore_index=True)

    X_train, y_train, train_ids = prepare_xy(labeled)

    X_train, y_train, train_ids = _oversample_positive(
        X=X_train,
        y=y_train,
        employee_ids=train_ids,
        seed=seed,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Duplicate Employee IDs in training data were detected.",
            category=UserWarning,
        )

        model = train_model(
            X_train,
            y_train,
            train_ids,
            seed=seed,
        )

    return model