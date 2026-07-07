"""
Provided framework utilities — do NOT modify this file.

Use call_oracle, train_model, and the data loaders from strategy.py.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

_STUDENT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _STUDENT_DIR / "constants.yaml"

with open(_CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MAX_LABELED = CONFIG["max_labeled"]
MAX_RUNTIME_SEC = CONFIG["max_runtime_sec"]
SEEDS = CONFIG["seeds"]
ID_COLUMN = "Employee ID"
TARGET_COLUMN = "Attrition"


class BudgetExceededError(Exception):
    """Raised when call_oracle would exceed the unique-ID budget."""


class OracleError(Exception):
    """Raised for invalid oracle requests."""


class _OracleTracker:
    def __init__(self, max_labeled: int):
        self.max_labeled = max_labeled
        self._queried: set[str] = set()

    def register(self, sample_ids: Iterable[str]) -> None:
        sample_ids = [str(sid) for sid in sample_ids]
        new_ids = set(sample_ids) - self._queried
        if len(self._queried) + len(new_ids) > self.max_labeled:
            raise BudgetExceededError(
                f"Oracle budget exceeded: {len(self._queried)} unique IDs already queried, "
                f"requested {len(new_ids)} new IDs, maximum allowed is {self.max_labeled}."
            )
        self._queried |= new_ids

    @property
    def queried_ids(self) -> set[str]:
        return set(self._queried)

    def reset(self) -> None:
        self._queried.clear()


_oracle_tracker = _OracleTracker(MAX_LABELED)
_pool_labels: dict[str, int] | None = None
_pool_df: pd.DataFrame | None = None
_pool_employee_ids: set[str] | None = None
_feature_columns: list[str] | None = None
_active_seed: int | None = None


def _data_path(relative_path: str) -> Path:
    return _STUDENT_DIR / relative_path


def _load_pool_labels() -> dict[str, int]:
    global _pool_labels
    if _pool_labels is None:
        labels_path = _data_path(CONFIG["data"]["pool_labels"])
        with open(labels_path, "rb") as f:
            _pool_labels = pickle.load(f)
    return _pool_labels


def _load_feature_columns() -> list[str]:
    global _feature_columns
    if _feature_columns is None:
        columns_path = _data_path(CONFIG["data"]["feature_columns"])
        with open(columns_path, "rb") as f:
            _feature_columns = pickle.load(f)
    return _feature_columns


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
    encoded = pd.get_dummies(features, drop_first=True)
    return encoded.reindex(columns=_load_feature_columns(), fill_value=0)


def set_active_seed(seed: int) -> None:
    """Set the active seed for the current evaluation run."""
    global _active_seed
    if seed not in SEEDS:
        raise ValueError(f"Invalid seed {seed}. Allowed seeds: {SEEDS}")
    _active_seed = seed


def get_active_seed() -> int:
    if _active_seed is None:
        raise RuntimeError("Active seed is not set. This function is called by the framework.")
    return _active_seed


def reset_oracle() -> None:
    """Reset oracle query history. Called by evaluation.py before each seed."""
    _oracle_tracker.reset()


def load_pool() -> pd.DataFrame:
    """Load the unlabeled candidate pool (features only, no Attrition column)."""
    global _pool_df, _pool_employee_ids
    if _pool_df is None:
        _pool_df = pd.read_csv(_data_path(CONFIG["data"]["pool"]))
        if TARGET_COLUMN in _pool_df.columns:
            raise RuntimeError("Pool data must not contain Attrition labels.")
        _pool_employee_ids = set(_pool_df[ID_COLUMN].astype(str))
    return _pool_df.copy()


def load_initial_labeled(seed: int) -> pd.DataFrame:
    """
    Load the initial labeled set for a seed.

    Returns a DataFrame with all pool feature columns plus:
      - Employee ID
      - Attrition encoded as int (1 = Left, 0 = Stayed)
    """
    if seed not in SEEDS:
        raise ValueError(f"Invalid seed {seed}. Allowed seeds: {SEEDS}")

    splits_path = _data_path(CONFIG["data"]["splits_template"].format(seed=seed))
    with open(splits_path, "rb") as f:
        split = pickle.load(f)

    pool = load_pool()
    employee_ids = [str(eid) for eid in split["employee_ids"]]
    labels = np.asarray(split["labels"], dtype=int)

    rows = pool[pool[ID_COLUMN].astype(str).isin(employee_ids)].copy()
    id_to_label = dict(zip(employee_ids, labels))
    rows[TARGET_COLUMN] = rows[ID_COLUMN].astype(str).map(id_to_label).astype(int)
    order = {eid: i for i, eid in enumerate(employee_ids)}
    rows["_order"] = rows[ID_COLUMN].astype(str).map(order)
    rows = rows.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return rows


def load_test(seed: int) -> pd.DataFrame:
    """
    Load the labeled test set for local evaluation.

    Returns a DataFrame with feature columns, Employee ID, and encoded Attrition.
    """
    if seed not in SEEDS:
        raise ValueError(f"Invalid seed {seed}. Allowed seeds: {SEEDS}")

    test_path = _data_path(CONFIG["data"]["test_template"].format(seed=seed))
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    df = test_data["features"].copy()
    df[ID_COLUMN] = test_data["employee_ids"]
    df[TARGET_COLUMN] = test_data["labels"]
    return df


def _get_test_employee_ids(seed: int) -> set[str]:
    test_path = _data_path(CONFIG["data"]["test_template"].format(seed=seed))
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return {str(eid) for eid in test_data["employee_ids"]}


def call_oracle(sample_ids: list[str] | np.ndarray) -> pd.DataFrame:
    """
    Request ground-truth labels for pool samples by Employee ID.

    Each unique ID counts toward the oracle budget (max_labeled in constants.yaml).
    Re-querying an already annotated ID returns the cached label without extra cost.
    """
    if _pool_employee_ids is None:
        load_pool()

    sample_ids = [str(sid) for sid in sample_ids]
    if len(sample_ids) == 0:
        return pd.DataFrame(columns=[ID_COLUMN, TARGET_COLUMN])

    unknown = set(sample_ids) - _pool_employee_ids
    if unknown:
        raise OracleError(
            f"Employee IDs not found in pool: {sorted(list(unknown))[:5]}"
            f"{'...' if len(unknown) > 5 else ''}"
        )

    _oracle_tracker.register(sample_ids)

    labels_map = _load_pool_labels()
    pool = load_pool()
    pool_id_str = pool[ID_COLUMN].astype(str)

    rows = []
    for sid in sample_ids:
        row = pool.loc[pool_id_str == sid].iloc[0].copy()
        row[TARGET_COLUMN] = int(labels_map[sid])
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Convert a labeled DataFrame to model-ready features, labels, and Employee IDs.

    Attrition must already be encoded as 0/1 or Stayed/Left strings.
    """
    work = df.copy()
    if work[TARGET_COLUMN].dtype == object:
        y = (work[TARGET_COLUMN] == "Left").astype(int).to_numpy()
    else:
        y = work[TARGET_COLUMN].astype(int).to_numpy()

    employee_ids = work[ID_COLUMN].astype(str).to_numpy()
    X = _encode_features(work)
    return X, y, employee_ids


def train_model(
    X: pd.DataFrame | np.ndarray,
    y: Iterable[int],
    employee_ids: Iterable[str],
    seed: int,
) -> RandomForestClassifier:
    """
    Train the provided RandomForestClassifier on labeled data.

    Raises if any Employee ID belongs to the test set for this seed.
    """
    if seed not in SEEDS:
        raise ValueError(f"Invalid seed {seed}. Allowed seeds: {SEEDS}")

    employee_ids = [str(eid) for eid in employee_ids]
    y = np.asarray(list(y), dtype=int)

    if len(employee_ids) != len(y):
        raise ValueError("employee_ids and y must have the same length.")
    if len(employee_ids) == 0:
        raise ValueError("Cannot train on an empty dataset.")

    test_ids = _get_test_employee_ids(seed)
    overlap = set(employee_ids) & test_ids
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(
            f"Training data contains test Employee IDs: {preview}"
            f"{'...' if len(overlap) > 5 else ''}"
        )

    if len(employee_ids) != len(set(employee_ids)):
        warnings.warn(
            "Duplicate Employee IDs in training data were detected.",
            stacklevel=2,
        )

    rf_cfg = CONFIG["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        random_state=seed,
        n_jobs=rf_cfg["n_jobs"],
    )
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestClassifier, seed: int) -> float:
    """Return F1 score for the Left class (positive label = 1) on the test set."""
    test_df = load_test(seed)
    X_test, y_test, _ = prepare_xy(test_df)
    preds = model.predict(X_test)
    return float(f1_score(y_test, preds, pos_label=1))


def get_oracle_usage() -> dict[str, int]:
    """Return oracle usage statistics for debugging."""
    return {
        "unique_queried": len(_oracle_tracker.queried_ids),
        "remaining": MAX_LABELED - len(_oracle_tracker.queried_ids),
    }
