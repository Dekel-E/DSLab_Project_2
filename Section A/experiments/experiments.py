"""
Local-only benchmarking harness for Section A — NOT submitted.

Compares active-learning strategy variants across seeds 1-3 and saves
results (CSV) + plots (PNG) for the video presentation.

Usage:
    python experiments.py curves    # selection strategies: F1 vs labels used
    python experiments.py dup       # positive-duplication ratio sweep (tiled)
    python experiments.py center    # margin center 0.5 vs lower centers
    python experiments.py pseudo    # pseudo-labeling on/off
    python experiments.py cv        # CV-chosen dup ratio vs test-optimal
    python experiments.py plots     # regenerate figures from stored CSVs
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

_SECTION_A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SECTION_A))  # make utils.py importable from this subfolder

import numpy as np
import pandas as pd

import utils
from utils import (
    call_oracle,
    evaluate_model,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    reset_oracle,
    set_active_seed,
    train_model,
)

warnings.filterwarnings("ignore")

SEEDS = [1, 2, 3]
BUDGET = 5000
RESULTS_DIR = _SECTION_A / "exp_results"
PLOTS_DIR = _SECTION_A / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- helpers

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode features using only the public prepare_xy API (dummy label)."""
    work = df.copy()
    work["Attrition"] = 0
    X, _, _ = prepare_xy(work)
    return X


def duplicate_positives(X, y, ids, dup: float):
    """Append `dup` extra copies of the positive rows (tiled, deterministic)."""
    if dup <= 0:
        return X, y, ids
    pos_idx = np.where(y == 1)[0]
    n_extra = int(round(dup * len(pos_idx)))
    if n_extra == 0:
        return X, y, ids
    reps = int(np.ceil(n_extra / len(pos_idx)))
    idx = np.tile(pos_idx, reps)[:n_extra]
    Xa = pd.concat([X, X.iloc[idx]], ignore_index=True)
    ya = np.concatenate([y, y[idx]])
    ida = np.concatenate([ids, ids[idx]])
    return Xa, ya, ida


def train(X, y, ids, seed):
    return train_model(X, y, ids, seed=seed)


def run_al(
    seed: int,
    selection: str = "uncertainty",   # 'random' | 'uncertainty' | 'mixed'
    batch_size: int = 500,
    center: float = 0.5,              # margin center for uncertainty
    random_frac: float = 0.2,         # only for 'mixed'
    eval_dup: float | None = None,    # if set, evaluate each checkpoint with this dup
    record_curve: bool = False,
):
    """Run one AL episode; return (labeled_df, curve list of (n_labels, f1))."""
    reset_oracle()
    set_active_seed(seed)
    rng = np.random.RandomState(seed)

    labeled = load_initial_labeled(seed)
    pool = load_pool()
    have = set(labeled["Employee ID"].astype(str))
    cand = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
    cand_ids = cand["Employee ID"].astype(str).to_numpy()
    X_cand = encode(cand)

    curve = []
    spent = 0
    while spent < BUDGET:
        k = min(batch_size, BUDGET - spent)
        X, y, ids = prepare_xy(labeled)
        model = train(X, y, ids, seed)

        if record_curve:
            if eval_dup:
                Xa, ya, ida = duplicate_positives(X, y, ids, eval_dup)
                curve.append((len(labeled), evaluate_model(train(Xa, ya, ida, seed), seed)))
            else:
                curve.append((len(labeled), evaluate_model(model, seed)))

        if selection == "random":
            take_idx = rng.choice(len(cand_ids), k, replace=False)
        else:
            proba = model.predict_proba(X_cand)[:, 1]
            margin = np.abs(proba - center)
            if selection == "mixed":
                k_unc = int(round(k * (1 - random_frac)))
                unc_idx = np.argsort(margin)[:k_unc]
                rest = np.setdiff1d(np.arange(len(cand_ids)), unc_idx)
                rnd_idx = rng.choice(rest, k - k_unc, replace=False)
                take_idx = np.concatenate([unc_idx, rnd_idx])
            else:
                take_idx = np.argsort(margin)[:k]

        new_rows = call_oracle(list(cand_ids[take_idx]))
        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        spent += k
        keep = np.setdiff1d(np.arange(len(cand_ids)), take_idx)
        cand_ids = cand_ids[keep]
        X_cand = X_cand.iloc[keep].reset_index(drop=True)

    if record_curve:
        X, y, ids = prepare_xy(labeled)
        if eval_dup:
            X, y, ids = duplicate_positives(X, y, ids, eval_dup)
        curve.append((len(labeled), evaluate_model(train(X, y, ids, seed), seed)))
    return labeled, curve


def final_f1(labeled, seed, dup=0.0):
    X, y, ids = prepare_xy(labeled)
    X, y, ids = duplicate_positives(X, y, ids, dup)
    return evaluate_model(train(X, y, ids, seed), seed)


def save(df: pd.DataFrame, name: str):
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved -> {path}")


# ---------------------------------------------------------------- experiments

def exp_curves():
    """Selection strategies, learning curves. Checkpoints evaluated with dup=1.0."""
    rows = []
    for seed in SEEDS:
        for sel in ["random", "uncertainty", "mixed"]:
            t0 = time.perf_counter()
            _, curve = run_al(seed, selection=sel, eval_dup=1.0, record_curve=True)
            dt = time.perf_counter() - t0
            for n, f1 in curve:
                rows.append({"seed": seed, "selection": sel, "n_labels": n, "f1": f1})
            print(f"seed={seed} sel={sel:11s} final F1={curve[-1][1]:.4f}  ({dt:.0f}s)")
    df = pd.DataFrame(rows)
    save(df, "curves")
    summary = df.groupby(["selection", "n_labels"])["f1"].mean().reset_index()
    print(summary.pivot(index="n_labels", columns="selection", values="f1").round(4))


def exp_dup():
    """Duplication ratio sweep on uncertainty-selected sets."""
    rows = []
    for seed in SEEDS:
        labeled, _ = run_al(seed, selection="uncertainty")
        for dup in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            f1 = final_f1(labeled, seed, dup)
            rows.append({"seed": seed, "dup": dup, "f1": f1})
            print(f"seed={seed} dup={dup:.2f}  F1={f1:.4f}")
    df = pd.DataFrame(rows)
    save(df, "dup_sweep")
    print(df.groupby("dup")["f1"].mean().round(4))


def exp_center():
    """Margin center for uncertainty selection (final F1 with dup sweep per center)."""
    rows = []
    for seed in SEEDS:
        for center in [0.5, 0.4, 0.35, 0.3]:
            labeled, _ = run_al(seed, selection="uncertainty", center=center)
            for dup in [0.75, 1.0, 1.25, 1.5]:
                f1 = final_f1(labeled, seed, dup)
                rows.append({"seed": seed, "center": center, "dup": dup, "f1": f1})
            best = max(r["f1"] for r in rows if r["seed"] == seed and r["center"] == center)
            print(f"seed={seed} center={center:.2f}  best F1={best:.4f}")
    df = pd.DataFrame(rows)
    save(df, "center")
    print(df.groupby(["center", "dup"])["f1"].mean().round(4).unstack())


def exp_pseudo():
    """Pseudo-labeling remaining unlabeled pool with confident predictions."""
    rows = []
    for seed in SEEDS:
        labeled, _ = run_al(seed, selection="uncertainty")
        pool = load_pool()
        have = set(labeled["Employee ID"].astype(str))
        rest = pool[~pool["Employee ID"].astype(str).isin(have)].reset_index(drop=True)
        X_rest = encode(rest)

        X, y, ids = prepare_xy(labeled)
        Xd, yd, idd = duplicate_positives(X, y, ids, 1.0)
        base_model = train(Xd, yd, idd, seed)
        base = evaluate_model(base_model, seed)
        rows.append({"seed": seed, "variant": "no_pseudo", "thresh": None, "f1": base})
        print(f"seed={seed} no_pseudo           F1={base:.4f}")

        proba = base_model.predict_proba(X_rest)[:, 1]
        for thresh in [0.85, 0.9, 0.95]:
            conf = (proba >= thresh) | (proba <= 1 - thresh)
            Xp = pd.concat([X, X_rest[conf]], ignore_index=True)
            yp = np.concatenate([y, (proba[conf] >= 0.5).astype(int)])
            idp = np.concatenate([ids, rest["Employee ID"].astype(str).to_numpy()[conf]])
            Xp, yp, idp = duplicate_positives(Xp, yp, idp, 1.0)
            f1 = evaluate_model(train(Xp, yp, idp, seed), seed)
            rows.append({"seed": seed, "variant": "pseudo", "thresh": thresh, "f1": f1})
            print(f"seed={seed} pseudo t={thresh:.2f} (+{conf.sum()}) F1={f1:.4f}")
    save(pd.DataFrame(rows), "pseudo")


def cv_pick_dup(labeled, seed, ratios=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    """Choose dup ratio by 3-fold CV on the labeled set only (no test access)."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    X, y, ids = prepare_xy(labeled)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {r: [] for r in ratios}
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, ytr, idtr = X.iloc[tr_idx], y[tr_idx], ids[tr_idx]
        Xva, yva = X.iloc[va_idx], y[va_idx]
        for r in ratios:
            Xa, ya, ida = duplicate_positives(Xtr, ytr, idtr, r)
            m = train(Xa, ya, ida, seed)
            scores[r].append(f1_score(yva, m.predict(Xva), pos_label=1))
    mean_scores = {r: float(np.mean(v)) for r, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


def exp_cv():
    """Does labeled-set CV pick a dup ratio close to the test-optimal one?"""
    rows = []
    for seed in SEEDS:
        labeled, _ = run_al(seed, selection="uncertainty")
        best_cv, cv_scores = cv_pick_dup(labeled, seed)
        test_scores = {r: final_f1(labeled, seed, r) for r in cv_scores}
        best_test = max(test_scores, key=test_scores.get)
        print(f"seed={seed}  CV pick dup={best_cv} (test F1={test_scores[best_cv]:.4f})"
              f"  test-optimal dup={best_test} (F1={test_scores[best_test]:.4f})")
        for r in cv_scores:
            rows.append({"seed": seed, "dup": r, "cv_f1": cv_scores[r], "test_f1": test_scores[r]})
    save(pd.DataFrame(rows), "cv_dup")


# ---------------------------------------------------------------- plots

def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. learning curves
    p = RESULTS_DIR / "curves.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for sel, g in df.groupby("selection"):
            m = g.groupby("n_labels")["f1"].mean()
            ax.plot(m.index, m.values, marker="o", label=sel)
        ax.set_xlabel("Labeled samples used")
        ax.set_ylabel("F1 (Left), mean over seeds 1-3")
        ax.set_title("Active-learning selection strategies (final model rebalanced)")
        ax.legend(title="Selection")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "learning_curves.png", dpi=150)
        print("saved plot: learning_curves.png")

    # 2. dup sweep
    p = RESULTS_DIR / "dup_sweep.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for seed, g in df.groupby("seed"):
            ax.plot(g["dup"], g["f1"], marker="o", alpha=0.5, label=f"seed {seed}")
        m = df.groupby("dup")["f1"].mean()
        ax.plot(m.index, m.values, marker="s", color="black", linewidth=2, label="mean")
        ax.set_xlabel("Extra copies of positive samples (duplication ratio)")
        ax.set_ylabel("F1 (Left)")
        ax.set_title("Class rebalancing via positive duplication")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "dup_sweep.png", dpi=150)
        print("saved plot: dup_sweep.png")

    # 3. final per-seed bars: skeleton vs random vs final strategy
    p = RESULTS_DIR / "final_bars.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        variants = df["variant"].unique()
        width = 0.8 / len(variants)
        xs = np.arange(len(SEEDS))
        for i, v in enumerate(variants):
            g = df[df["variant"] == v].set_index("seed").loc[SEEDS, "f1"]
            ax.bar(xs + i * width, g.values, width, label=v)
        ax.set_xticks(xs + width * (len(variants) - 1) / 2)
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_ylabel("F1 (Left)")
        ax.set_title("Final F1 per seed by strategy")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "final_bars.png", dpi=150)
        print("saved plot: final_bars.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "curves"
    {
        "curves": exp_curves,
        "dup": exp_dup,
        "center": exp_center,
        "pseudo": exp_pseudo,
        "cv": exp_cv,
        "plots": make_plots,
    }[cmd]()
