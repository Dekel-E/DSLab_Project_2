"""
Local-only improvement experiments (round 2) — NOT submitted.

Usage:
    python experiments2.py batch     # batch size 250 / 500 / 1000
    python experiments2.py variants  # rebalanced-loop, diversity, pos-harvest selection
    python experiments2.py noise     # OOF label-noise filtering before final fit
    python experiments2.py cvgrid    # finer dup grid + repeated CV stability
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiments import (
    BUDGET,
    SEEDS,
    duplicate_positives,
    encode,
    save,
    train,
)
from utils import (
    call_oracle,
    evaluate_model,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    reset_oracle,
    set_active_seed,
)

warnings.filterwarnings("ignore")


def run_al2(
    seed: int,
    batch_size: int = 500,
    loop_dup: float = 0.0,        # duplication applied to the selection model
    diversity: bool = False,      # cluster-diverse batches
    pos_harvest_frac: float = 0.0,  # fraction of batch = highest P(Left)
):
    reset_oracle()
    set_active_seed(seed)
    labeled = load_initial_labeled(seed)
    pool = load_pool()
    have = set(labeled["Employee ID"].astype(str))
    cand = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
    cand_ids = cand["Employee ID"].astype(str).to_numpy()
    X_cand = encode(cand)

    spent = 0
    while spent < BUDGET:
        k = min(batch_size, BUDGET - spent)
        X, y, ids = prepare_xy(labeled)
        if loop_dup > 0:
            Xd, yd, idd = duplicate_positives(X, y, ids, loop_dup)
            model = train(Xd, yd, idd, seed)
        else:
            model = train(X, y, ids, seed)

        proba = model.predict_proba(X_cand)[:, 1]
        margin = np.abs(proba - 0.5)

        k_pos = int(round(k * pos_harvest_frac))
        k_unc = k - k_pos

        if diversity:
            top = np.argsort(margin, kind="stable")[: min(4 * k_unc, len(cand_ids))]
            n_cl = min(k_unc, len(top))
            km = MiniBatchKMeans(n_clusters=n_cl, random_state=seed, n_init=3)
            cl = km.fit_predict(X_cand.iloc[top])
            take = []
            for c in range(n_cl):
                members = top[cl == c]
                if len(members):
                    take.append(members[np.argmin(margin[members])])
            unc_idx = np.array(take[:k_unc], dtype=int)
            # backfill if clustering returned fewer
            if len(unc_idx) < k_unc:
                extra = [i for i in np.argsort(margin, kind="stable") if i not in set(unc_idx)]
                unc_idx = np.concatenate([unc_idx, np.array(extra[: k_unc - len(unc_idx)], dtype=int)])
        else:
            unc_idx = np.argsort(margin, kind="stable")[:k_unc]

        if k_pos > 0:
            rest = np.setdiff1d(np.arange(len(cand_ids)), unc_idx)
            pos_idx = rest[np.argsort(-proba[rest], kind="stable")[:k_pos]]
            take_idx = np.concatenate([unc_idx, pos_idx])
        else:
            take_idx = unc_idx

        new_rows = call_oracle(list(cand_ids[take_idx]))
        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        spent += len(take_idx)
        keep = np.setdiff1d(np.arange(len(cand_ids)), take_idx)
        cand_ids = cand_ids[keep]
        X_cand = X_cand.iloc[keep].reset_index(drop=True)

    return labeled


def eval_with_dup(labeled, seed, dup):
    X, y, ids = prepare_xy(labeled)
    Xa, ya, ida = duplicate_positives(X, y, ids, dup)
    return evaluate_model(train(Xa, ya, ida, seed), seed)


def best_over_dups(labeled, seed, dups=(1.0, 1.5, 2.0, 2.5)):
    return max((eval_with_dup(labeled, seed, d), d) for d in dups)


def exp_batch():
    rows = []
    for seed in SEEDS:
        for bs in [250, 500, 1000]:
            t0 = time.perf_counter()
            labeled = run_al2(seed, batch_size=bs)
            f1, d = best_over_dups(labeled, seed)
            rows.append({"seed": seed, "batch": bs, "f1": f1, "best_dup": d,
                         "al_sec": round(time.perf_counter() - t0, 1)})
            print(f"seed={seed} batch={bs:5d}  best F1={f1:.4f} (dup={d})  {rows[-1]['al_sec']}s")
    df = pd.DataFrame(rows)
    save(df, "batch_size")
    print(df.groupby("batch")["f1"].mean().round(4))


def exp_variants():
    rows = []
    variants = {
        "baseline": dict(),
        "loop_dup_1.0": dict(loop_dup=1.0),
        "diversity": dict(diversity=True),
        "pos_harvest_0.2": dict(pos_harvest_frac=0.2),
    }
    for seed in SEEDS:
        for name, kw in variants.items():
            labeled = run_al2(seed, **kw)
            f1, d = best_over_dups(labeled, seed)
            rows.append({"seed": seed, "variant": name, "f1": f1, "best_dup": d})
            print(f"seed={seed} {name:16s} best F1={f1:.4f} (dup={d})")
    df = pd.DataFrame(rows)
    save(df, "variants2")
    print(df.groupby("variant")["f1"].mean().round(4))


def exp_noise():
    """Remove rows whose 3-fold OOF probability strongly contradicts the label."""
    rows = []
    for seed in SEEDS:
        labeled = run_al2(seed)
        X, y, ids = prepare_xy(labeled)
        base_model_cls = train(X, y, ids, seed)  # just to reuse config
        oof = cross_val_predict(
            base_model_cls, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        for name, thresh in [("no_filter", None), ("t0.80", 0.80), ("t0.90", 0.90), ("t0.95", 0.95)]:
            if thresh is None:
                keep = np.ones(len(y), bool)
            else:
                wrong = np.where(y == 1, oof < 1 - thresh, oof > thresh)
                keep = ~wrong
            for dup in [1.5, 2.0, 2.5]:
                Xa, ya, ida = duplicate_positives(X[keep], y[keep], ids[keep], dup)
                f1 = evaluate_model(train(Xa, ya, ida, seed), seed)
                rows.append({"seed": seed, "filter": name, "dup": dup, "f1": f1,
                             "removed": int((~keep).sum())})
            best = max(r["f1"] for r in rows if r["seed"] == seed and r["filter"] == name)
            print(f"seed={seed} {name:10s} removed={int((~keep).sum()):4d}  best F1={best:.4f}")
    df = pd.DataFrame(rows)
    save(df, "noise_filter")
    print(df.groupby(["filter", "dup"])["f1"].mean().round(4).unstack())


def exp_cvgrid():
    """Repeated CV over a finer dup grid: does the pick beat single-CV / fixed 2.0?"""
    ratios = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
    rows = []
    for seed in SEEDS:
        labeled = run_al2(seed)
        X, y, ids = prepare_xy(labeled)
        scores = {r: [] for r in ratios}
        for rep in range(2):
            skf = StratifiedKFold(3, shuffle=True, random_state=seed + 100 * rep)
            for tr, va in skf.split(X, y):
                for r in ratios:
                    Xa, ya, ida = duplicate_positives(X.iloc[tr], y[tr], ids[tr], r)
                    m = train(Xa, ya, ida, seed)
                    scores[r].append(f1_score(y[va], m.predict(X.iloc[va]), pos_label=1))
        mean_scores = {r: float(np.mean(v)) for r, v in scores.items()}
        pick = max(mean_scores, key=mean_scores.get)
        test_scores = {r: eval_with_dup(labeled, seed, r) for r in ratios}
        best_test = max(test_scores, key=test_scores.get)
        print(f"seed={seed} repCV pick={pick} (test {test_scores[pick]:.4f}) | fixed2.0 {test_scores[2.0]:.4f}"
              f" | oracle-best={best_test} ({test_scores[best_test]:.4f})")
        for r in ratios:
            rows.append({"seed": seed, "dup": r, "cv_f1": mean_scores[r], "test_f1": test_scores[r]})
    save(pd.DataFrame(rows), "cvgrid")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "batch"
    {
        "batch": exp_batch,
        "variants": exp_variants,
        "noise": exp_noise,
        "cvgrid": exp_cvgrid,
    }[cmd]()
