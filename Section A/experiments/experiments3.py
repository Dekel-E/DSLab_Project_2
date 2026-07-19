"""
Local-only experiments (round 3) — lecture-inspired variants. NOT submitted.

Tests the two Lecture-5 selection strategies we had not yet tried:
  qbc      Query-by-Committee: 3 RFs (seeds 1/2/3), disagreement = std of P(Left)
  density  Density-weighted uncertainty (Settles): uncertainty x similarity^beta

Usage: python experiments3.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for utils.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from experiments import BUDGET, SEEDS, duplicate_positives, encode, save, train
from experiments2 import best_over_dups
from utils import (
    call_oracle,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    reset_oracle,
    set_active_seed,
)

warnings.filterwarnings("ignore")


def run_al3(seed: int, selection: str = "uncertainty", batch_size: int = 500, beta: float = 1.0):
    reset_oracle()
    set_active_seed(seed)
    labeled = load_initial_labeled(seed)
    pool = load_pool()
    have = set(labeled["Employee ID"].astype(str))
    cand = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
    cand_ids = cand["Employee ID"].astype(str).to_numpy()
    X_cand = encode(cand)

    density = None
    if selection == "density":
        # similarity to the candidate cloud: inverse mean distance to 10 NN
        Xs = StandardScaler().fit_transform(X_cand.astype(float))
        nn = NearestNeighbors(n_neighbors=11).fit(Xs)
        dist, _ = nn.kneighbors(Xs)
        density = 1.0 / (1.0 + dist[:, 1:].mean(axis=1))  # skip self

    spent = 0
    while spent < BUDGET:
        k = min(batch_size, BUDGET - spent)
        X, y, ids = prepare_xy(labeled)

        if selection == "qbc":
            probas = [train(X, y, ids, s).predict_proba(X_cand)[:, 1] for s in (1, 2, 3)]
            score = np.std(np.vstack(probas), axis=0)  # disagreement, higher = better
            take_idx = np.argsort(-score, kind="stable")[:k]
        else:
            model = train(X, y, ids, seed)
            proba = model.predict_proba(X_cand)[:, 1]
            unc = 1.0 - 2.0 * np.abs(proba - 0.5)  # in [0,1], higher = more uncertain
            if selection == "density":
                score = unc * (density ** beta)
            else:
                score = unc
            take_idx = np.argsort(-score, kind="stable")[:k]

        new_rows = call_oracle(list(cand_ids[take_idx]))
        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        spent += k
        keep = np.setdiff1d(np.arange(len(cand_ids)), take_idx)
        cand_ids = cand_ids[keep]
        X_cand = X_cand.iloc[keep].reset_index(drop=True)
        if density is not None:
            density = density[keep]

    return labeled


if __name__ == "__main__":
    rows = []
    for seed in SEEDS:
        for sel, kw in [("uncertainty", {}), ("qbc", {}),
                        ("density_b1.0", {"selection": "density", "beta": 1.0}),
                        ("density_b0.5", {"selection": "density", "beta": 0.5})]:
            t0 = time.perf_counter()
            labeled = run_al3(seed, **({"selection": sel} if not kw else kw))
            f1, d = best_over_dups(labeled, seed)
            dt = time.perf_counter() - t0
            rows.append({"seed": seed, "selection": sel, "f1": f1, "best_dup": d})
            print(f"seed={seed} {sel:13s} best F1={f1:.4f} (dup={d})  {dt:.0f}s")
    df = pd.DataFrame(rows)
    save(df, "lecture_variants")
    print(df.groupby("selection")["f1"].mean().round(4))
