# Section A — Video Script (≤100 s, 6 slides, 2 speakers)

**Budget:** 234 spoken words ≈ **94 s** at a normal 150 wpm pace — ~6 s of headroom.
**Speakers:** A = slides 1–3, B = slides 4–6. Both speak ~47 s. (Rubric: 3 pts for meaningful participation from both.)
**Slides:** exactly 6. Every slide is one full-bleed figure from `plots/` + a one-line title. No bullet walls — the figures are already self-interpreting (rubric: 7.5 pts).

---

## Slide 1 — Problem & constraints · 0:00–0:12 · Speaker A

**Visual:** no plot. A constraints strip:
`500 free labels → oracle: 5,000 queries → fixed RandomForest → metric: F1(Left), pool prior ⅓`
plus a tiny 3-box diagram: `label → retrain → select`.

> "Our task: predict which employees leave. Five hundred free labels, a five-thousand-query oracle, a fixed Random Forest, and one metric — F1 on the minority *Left* class. First question we asked: where's the ceiling?"

*(30 words · 12 s)*

---

## Slide 2 — Diagnosis · 0:12–0:30 · Speaker A

**Visual:** `plots/fig1_diagnosis.png` — F1 vs decision threshold, RF trained on all 14,900 pool labels.
**Title:** *"The metric is the bottleneck, not the data."*

> "We trained on all fourteen-thousand-nine-hundred pool labels. F1 was only 0.586 — because `predict` uses a fixed 0.5 threshold. Sweeping the threshold gives 0.657 at 0.38. So seven points sit in the threshold, not in the labels. That reframed the whole project."

*(42 words · 18 s)*

**Cue:** point at the two dots as you say 0.586 and 0.657.

---

## Slide 3 — Querying · 0:30–0:45 · Speaker A

**Visual:** `plots/fig2_learning_curves.png` — uncertainty vs mixed vs random.
**Title:** *"Uncertainty sampling buys ~2,500 labels."*

> "Querying first. Ten batches of five hundred: retrain, score the pool, take the samples closest to P equals 0.5. Uncertainty beats random by 0.021 and saturates near two-and-a-half thousand labels — random needs all five thousand to catch up."

*(38 words · 15 s)*

**Handoff:** "…which is why the second half of our budget went into something else." → hand to B.

---

## Slide 4 — Rebalancing · 0:45–1:03 · Speaker B

**Visual:** `plots/fig3_dup_sweep.png` (main). Inset or corner thumbnail: `plots/fig6_cv_adaptivity.png`.
**Title:** *"Rebalancing: the single largest lever (+0.05)."*

> "The big lever. We can't move the 0.5 threshold — but duplicating positive rows does it for us: plus 0.052. The curve is a broad plateau, so instead of hard-coding the ratio, repeated three-fold CV on *labeled data only* picks it per seed — and it tracks the test optimum."

*(45 words · 18 s)*

**Cue:** stress "labeled data only" — that's the no-leakage claim. Point at the inset on "tracks the test optimum."

---

## Slide 5 — What did *not* work · 1:03–1:21 · Speaker B

**Visual:** `plots/fig4_what_we_tried.png` — every idea, blue helped / red hurt.
**Title:** *"Everything we tried, in one picture."*

> "Most of what we tried failed — informatively. Querying near the optimal 0.38 threshold cost up to 0.096: those points are confident-stay noise. Query-by-committee lost 0.009 — a forest is already a committee. Pseudo-labeling and noise filtering did nothing: the ceiling is real ambiguity."

*(41 words · 17 s)*

**Cue:** sweep a finger down the red bars while speaking. Don't read the labels aloud — the figure does that.

---

## Slide 6 — Results & robustness · 1:21–1:36 · Speaker B

**Visual:** `plots/fig7_final_bars.png` (per-seed, 4 stages). Optional swap: `fig5_cumulative_gains.png` if you prefer the ceiling line visible.
**Title:** *"0.41 → 0.642, on every seed."*

> "End to end: 0.41 to 0.642 mean F1 — within 0.015 of the noise ceiling, and every stage improves every seed. Twelve seconds per run, exactly five thousand queries, bit-identical across two scikit-learn versions. The one component that wasn't — KMeans diversity batches — we deleted."

*(45 words · 18 s)*

---

## Totals

| Slide | Speaker | Words | Time |
|---|---|---|---|
| 1 Problem | A | 30 | 12 s |
| 2 Diagnosis | A | 42 | 18 s |
| 3 Querying | A | 38 | 15 s |
| 4 Rebalancing | B | 45 | 18 s |
| 5 Failures | B | 41 | 17 s |
| 6 Results | B | 45 | 18 s |
| **Total** | | **241** | **98 s** |

If a rehearsal runs long, cut in this order: (1) "and every stage improves every seed" on slide 6, (2) "That reframed the whole project" on slide 2, (3) "Query-by-committee lost 0.009 — a forest is already a committee" on slide 5.

## Recording checklist

- [ ] Both speakers audible and clearly distinct — A does 1–3, B does 4–6.
- [ ] Exactly 6 slides, screen-recorded at readable resolution (figures are 1650 px wide; don't downscale below 1280).
- [ ] Every figure keeps its legend and axis labels — do not crop them out.
- [ ] Final runtime **< 1:40**. Time the recording, not the rehearsal.
- [ ] Upload, set link sharing to "anyone with the link", paste the single URL into `video_link.txt`.
- [ ] Submit zip `<ID1_ID2>` containing **only** `strategy.py` + `video_link.txt` (Section A) and `gnn.py` (Section B).
