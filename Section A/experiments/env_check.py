"""Cross-environment stability check — run under multiple Python/sklearn versions.

For each seed and selection variant, prints a hash of the queried Employee IDs
(does the environment change WHICH samples get queried?), the CV-picked dup
ratio, and the final test F1.

Usage: python experiments/env_check.py
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from experiments import duplicate_positives, train
from experiments2 import run_al2
from utils import evaluate_model, prepare_xy

RATIOS = (1.25, 1.5, 1.75, 2.0, 2.25, 2.5)


def repcv_pick(X, y, ids, seed, reps=2):
    scores = {r: [] for r in RATIOS}
    for rep in range(reps):
        skf = StratifiedKFold(3, shuffle=True, random_state=seed + 100 * rep)
        for tr, va in skf.split(X, y):
            for r in RATIOS:
                Xa, ya, ida = duplicate_positives(X.iloc[tr], y[tr], ids[tr], r)
                m = train(Xa, ya, ida, seed)
                scores[r].append(f1_score(y[va], m.predict(X.iloc[va]), pos_label=1))
    return max(RATIOS, key=lambda r: float(np.mean(scores[r])))


print(f"python {sys.version.split()[0]} | sklearn {sklearn.__version__}")
for variant, kw in [("uncertainty", {}), ("diversity", {"diversity": True})]:
    for seed in [1, 2, 3]:
        labeled = run_al2(seed, **kw)
        ids_sorted = ",".join(sorted(labeled["Employee ID"].astype(str)))
        digest = hashlib.md5(ids_sorted.encode()).hexdigest()[:10]
        X, y, ids = prepare_xy(labeled)
        pick = repcv_pick(X, y, ids, seed)
        Xa, ya, ida = duplicate_positives(X, y, ids, pick)
        f1 = evaluate_model(train(Xa, ya, ida, seed), seed)
        print(f"{variant:12s} seed={seed}  ids_hash={digest}  cv_pick={pick}  F1={f1:.4f}")
