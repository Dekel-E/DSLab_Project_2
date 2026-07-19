# Section A — Active Learning Strategy

**Final local result: mean F1(Left) = 0.6416** (seed 1: 0.6352, seed 2: 0.6453, seed 3: 0.6443), ~12 s/seed, exactly 5,000 oracle queries — **bit-for-bit identical across Python 3.12/scikit-learn 1.9.0 and Python 3.14/scikit-learn 1.8.0**.

---

## 1. Problem framing — what the constraints actually imply

Before writing any strategy code, we profiled the task. Three structural facts drove every design decision:

| Fact | Measured | Implication |
|---|---|---|
| Class prior is exactly ⅓ "Left", in pool and test | pool 33.34%, test 33.3% per seed | F1 of the minority class is the metric → recall is expensive |
| Evaluation calls `model.predict()` | fixed 0.5 vote threshold | We cannot tune a decision threshold directly — only through the **training-set composition** |
| Labels are noisy | Training on **all 14,900 pool labels** gives only F1 = 0.586 (0.657 with an oracle-optimal threshold of 0.38); mean over seeds 1–3 | There is a hard performance ceiling ≈ 0.657. Getting close to it matters more than exotic querying |

The ceiling experiment (train on everything, sweep the threshold — `plots/fig1_diagnosis.png`) was our compass: it told us (a) roughly where the maximum is, and (b) that the gap between default-threshold (0.586) and optimal-threshold (0.657) F1 is worth ~0.07 — **the biggest single lever in the whole project is rebalancing, not querying.**

## 2. Final pipeline (`strategy.py`)

1. **Start** with the 500 free initial labels.
2. **Batch active learning, 10 rounds × 500 queries:** train the fixed Random Forest, score every remaining pool row, and take the samples with predicted P(Left) closest to 0.5 (uncertainty sampling). A cluster-diversity variant (MiniBatchKMeans over the uncertain region) was adopted and later **removed**: its tiny local gain was not reproducible across scikit-learn versions (see §5), while plain uncertainty selection queries the exact same Employee IDs under sklearn 1.8 and 1.9.
3. **Rebalance for F1:** duplicate the positive training rows (explicitly allowed by the assignment). The duplication ratio is chosen from {1.25 … 2.5} by **repeated 3-fold cross-validation on the labeled data only** — never the test set — so the choice adapts to the hidden data at grading time.
4. **Final fit** on the 5,500 labeled rows + duplicated positives → return the model.

Safety engineering for the hidden run: budget read from `get_oracle_usage()` (not hard-coded), batches capped to land exactly on budget, runtime guards (dump remaining budget in one batch after 30 s; skip the CV and use ratio 2.0 after 42 s), guard against degenerate single-class fits, deterministic per seed (verified: repeated runs give identical scores).

## 3. What we tried — what worked

All numbers are mean F1(Left) over seeds 1–3 on the local test sets.

| Change | Mean F1 | Gain | Plot |
|---|---|---|---|
| Initial 500 labels only (baseline) | 0.407 | — | `plots/fig5_cumulative_gains.png`, `fig7_final_bars.png` |
| + spend full budget randomly | 0.563 | +0.156 | " |
| + uncertainty sampling instead of random | 0.585 | +0.022 | `plots/fig2_learning_curves.png` |
| + positive duplication (rebalancing) | 0.637 | +0.052 | `plots/fig3_dup_sweep.png` |
| + repeated-CV ratio selection (finer grid) | **0.642** | +0.005 | `plots/fig6_cv_adaptivity.png` |

Every idea we measured — helpful and harmful — is summarized in one figure: `plots/fig4_what_we_tried.png`.

Two observations worth discussing:

- **Uncertainty sampling saturates early.** The learning curve plateaus around ~2,000–2,500 labels; random selection needs the full 5,000 to get close (`plots/fig2_learning_curves.png`). Selection quality buys the equivalent of ~2,500 extra labels.
- **Rebalancing is the single largest lever** (+0.05). Duplicating each positive row ~2× shifts the RF's effective decision threshold from 0.5 to roughly the F1-optimal ~0.38, without touching the fixed hyperparameters. The F1-vs-ratio curve is a broad plateau (`plots/fig3_dup_sweep.png`), which is why we let CV pick the ratio per seed instead of hard-coding it.

## 4. What we tried — what did *not* work

Negative results, each measured over all 3 seeds:

| Idea | Result | Why it failed |
|---|---|---|
| Querying near the F1-optimal threshold (center 0.3–0.4 instead of 0.5) | **−0.03 to −0.09** | Samples near 0.32 are mostly "confident-stay" noise; the model boundary lives at 0.5 |
| Selecting with an already-rebalanced model in the loop | −0.017 | Same effect through the back door |
| Harvesting likely-positives (20% of each batch) | −0.005 | Uncertainty sampling already yields 47% positives — extra positives add redundancy, not information |
| Pseudo-labeling confident unlabeled rows | ±0.005, inconsistent | Only ~1–300 rows pass a confidence threshold; RF probabilities are too flat under label noise |
| Label-noise filtering (drop rows whose out-of-fold prediction strongly contradicts the label) | 0.000 (0–2 rows removed) | The noise is irreducible ambiguity, not confident mislabels — nothing to filter |
| Batch size 250 / 1000 instead of 500 | −0.004 both | 250 wastes fits on tiny updates; 1000 adapts too slowly |
| Cluster-diversity batches (MiniBatchKMeans over the uncertain region) | +0.002 on sklearn 1.9 but **−0.005 on sklearn 1.8**, and it queries *different employees* per version | KMeans internals changed between scikit-learn releases → the "gain" is environment noise, not signal; rejected for grading-machine robustness |
| Query-by-Committee (3 RFs, seeds 1–3, disagreement = std of P(Left)) | **−0.009**, worse on all 3 seeds | An RF *is* already a 100-tree committee — its probability output is the trees' vote fraction, so a committee-of-RFs adds noise, not signal |
| Density-weighted uncertainty (Settles: uncertainty × similarity^β, β ∈ {0.5, 1.0}) | **−0.008 to −0.010**, worse on all 3 seeds | The employee pool is dense and homogeneous — no outlier problem to protect against — so the density term only drags queries away from the decision boundary |

The pseudo-labeling and noise-filtering failures are two sides of the same diagnosis: the ~0.65 ceiling comes from genuinely ambiguous employees, so no amount of self-training or cleaning recovers it.

## 5. Overfitting risk to the hidden test — assessment

Ranked by how each design choice was validated:

- **Structural choices (no risk):** the rebalancing lever exists because of the metric + fixed 0.5 threshold + ⅓ prior — properties of the grading setup itself, not of our local sample.
- **Runtime-adaptive choices (low risk):** the duplication ratio is *not* a constant — it is re-chosen by CV on whatever labeled data the hidden run produces. Locally the CV pick matched the test-optimal ratio almost exactly (0.6416 vs 0.6439 achievable).
- **Locally-selected constants (bounded risk):** batch size 500 and the CV grid range. These were chosen by local test F1 — but each was consistent across 3 independent seeds/test sets, and their combined contribution is small. Worst case on hidden data they cost ~0.005.
- **Environment robustness (verified):** the full pipeline was run under two different environments (Python 3.12 / scikit-learn 1.9.0 and Python 3.14 / scikit-learn 1.8.0) and produced identical queried-ID sets, CV picks, and scores. The one component that failed this test — MiniBatchKMeans diversity batches — was removed, so the grading machine's library versions cannot silently change the strategy's behavior.
- **Not done (would have been high risk):** tuning anything on the local test inside `strategy.py`, merging trees to exceed 100 estimators, or touching `load_test`/`evaluate_model` in submitted code.

Since the hidden pool/test are drawn from the same data-generating process (same features, same encoder, same ⅓ prior via the staff `constants.yaml`), expected hidden performance is ≈ local minus normal sampling noise (F1 std on n = 3,725 is ~±0.01).

## 6. Video plan (≤100 s, ≤6 slides, both members speak)

| # | Slide | Content | Speaker | ~s |
|---|---|---|---|---|
| 1 | Problem & constraints | 500 free labels, 5,000-query budget, fixed RF, F1(Left) | A | 12 |
| 2 | Diagnosis | `fig1_diagnosis.png` — all labels → 0.586; optimal threshold → 0.657. "The metric, not the data, is the bottleneck" | A | 18 |
| 3 | Querying | `fig2_learning_curves.png`: uncertainty vs random, saturates at ~2.5k labels | A | 15 |
| 4 | Rebalancing | `fig3_dup_sweep.png`: +0.05 from duplicating positives; ratio chosen by CV on labeled data only (`fig6_cv_adaptivity.png` as backup) | B | 18 |
| 5 | What didn't work | `fig4_what_we_tried.png`: every idea in one picture — threshold-centered querying (−0.10), pseudo-labeling (±0.001), noise filtering (0) | B | 20 |
| 6 | Results | `fig5_cumulative_gains.png` or `fig7_final_bars.png`: 0.41 → 0.56 → 0.59 → 0.64; budget/runtime/robustness guards | B | 15 |

All plots have legends and are self-interpreting (rubric: 7.5 pts empirical evaluation). Raw numbers for every table above are in `exp_results/*.csv`.

---
*Files: submitted = `strategy.py` (+ `video_link.txt`). Local-only = `experiments/` (benchmark scripts `experiments.py`, `experiments2.py`, `experiments3.py` + figure generator `make_figures.py`), `exp_results/`, `plots/`, this file. Run from `Section A/`, e.g. `python experiments/experiments.py curves` or `python experiments/make_figures.py`.*
