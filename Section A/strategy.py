"""
Student implementation file — submit this file only.

Implement run_active_learning(seed) to run your active learning strategy and return
a trained RandomForestClassifier. You may add helper functions in this file only.

Allowed imports: numpy, pandas, sklearn, scipy, collections, warnings, typing, utils
"""

from __future__ import annotations

from utils import (
    load_initial_labeled,
    prepare_xy,
    train_model,
)


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
    # --- TODO: implement your active learning strategy below ---

    # 1. Start from the provided initial labeled set (500 samples, free budget).
    labeled = load_initial_labeled(seed)

    # 2. Iteratively select Employee IDs from the pool, call_oracle(ids), and retrain.
    #    You may also add pseudo-labeled or augmented samples to training without
    #    using the oracle, as long as you respect the oracle budget enforced by utils.

    # Example skeleton (replace with your strategy):
    X_train, y_train, train_ids = prepare_xy(labeled)
    model = train_model(X_train, y_train, train_ids, seed=seed)

    return model
