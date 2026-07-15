"""
Render the video/report figures from exp_results/*.csv — local only, NOT submitted.

Usage: python make_figures.py
Writes PNGs to plots/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RES = HERE / "exp_results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

# ---- palette (validated reference palette, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # categorical slot 1 / positive pole
GREEN = "#008300"    # categorical slot 2
RED = "#e34948"      # negative pole (diverging)
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue
RAMP4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 10.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "legend.frameon": False,
})

FIGSIZE = (8.4, 4.7)
DPI = 200


def new_ax(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.text(0.06, 0.955, title, fontsize=14, fontweight="semibold", color=INK, ha="left")
    fig.text(0.06, 0.905, subtitle, fontsize=10, color=INK2, ha="left")
    fig.subplots_adjust(top=0.83, left=0.09, right=0.955, bottom=0.13)
    return fig, ax


def finish(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"saved plot: {name}")


# ------------------------------------------------- fig 1: diagnosis
def fig_diagnosis():
    p = RES / "threshold_sweep.csv"
    if not p.exists():  # compute once: all-labels model, threshold sweep, mean over seeds
        import pickle
        from sklearn.metrics import f1_score
        from utils import load_pool, load_test, prepare_xy, set_active_seed, train_model

        pool = load_pool()
        with open(HERE / "data/.pool_labels.pkl", "rb") as f:
            pl = pickle.load(f)
        full = pool.copy()
        full["Attrition"] = full["Employee ID"].astype(str).map(pl).astype(int)
        Xf, yf, idf = prepare_xy(full)
        rows = []
        ts = np.round(np.arange(0.10, 0.72, 0.02), 2)
        for seed in [1, 2, 3]:
            set_active_seed(seed)
            m = train_model(Xf, yf, idf, seed=seed)
            Xt, yt, _ = prepare_xy(load_test(seed))
            proba = m.predict_proba(Xt)[:, 1]
            for t in ts:
                rows.append({"seed": seed, "t": t,
                             "f1": f1_score(yt, (proba >= t).astype(int), pos_label=1)})
        pd.DataFrame(rows).to_csv(p, index=False)

    df = pd.read_csv(p)
    m = df.groupby("t")["f1"].mean()
    fig, ax = new_ax(
        "Diagnosis: the metric is the bottleneck, not the data",
        "Random Forest trained on ALL 14,900 pool labels · F1(Left) on local test vs decision threshold · mean of seeds 1–3",
    )
    ax.plot(m.index, m.values, color=BLUE, lw=2, solid_capstyle="round", zorder=3)

    t_best = float(m.idxmax())
    gain = m.max() - m.loc[0.50]
    for t, dy, label in [(0.50, 0.018, f"default predict() = 0.5\nF1 = {m.loc[0.50]:.3f}"),
                         (t_best, 0.012, f"best threshold = {t_best:.2f}\nF1 = {m.max():.3f}")]:
        ax.scatter([t], [m.loc[round(t, 2)]], s=64, color=BLUE, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(label, (t, m.loc[round(t, 2)]), xytext=(t + 0.015, m.loc[round(t, 2)] + dy),
                    fontsize=9.5, color=INK2, ha="left")
    ax.annotate("", xy=(t_best + 0.012, m.max() - 0.006), xytext=(0.493, m.loc[0.50] - 0.004),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(0.425, 0.505, f"+{gain:.3f} available from\nthreshold shifting alone",
            fontsize=9.5, color=INK2, ha="center")
    ax.set_xlabel("Decision threshold on P(Left)")
    ax.set_ylabel("F1 (Left)")
    ax.set_xlim(0.09, 0.73)
    finish(fig, "fig1_diagnosis.png")


# ------------------------------------------------- fig 2: learning curves
def fig_curves():
    df = pd.read_csv(RES / "curves.csv")
    m = df.groupby(["selection", "n_labels"])["f1"].mean().reset_index()
    fig, ax = new_ax(
        "Uncertainty sampling buys ~2,500 labels",
        "F1(Left) vs labels used, final model rebalanced ×1.0 · mean of seeds 1–3 · budget = 500 free + 5,000 queries",
    )
    series = [("uncertainty", BLUE, "uncertainty"),
              ("mixed", GREEN, "mixed (80/20 random)"),
              ("random", MUTED, "random")]
    for key, color, label in series:
        g = m[m["selection"] == key]
        lw = 2 if key != "random" else 1.6
        ax.plot(g["n_labels"], g["f1"], color=color, lw=lw, solid_capstyle="round",
                marker="o", ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=label, zorder=3 if key == "uncertainty" else 2)
        end = g.iloc[-1]
        ax.annotate(f"{label}  {end['f1']:.3f}", (end["n_labels"], end["f1"]),
                    xytext=(8, {"uncertainty": 5, "mixed": -11, "random": -4}[key]),
                    textcoords="offset points", fontsize=9.5, color=INK2, va="center")
    ax.axvline(2500, color=GRID, lw=1)
    ax.text(2540, 0.535, "plateau ≈ 2,500 labels", fontsize=9, color=MUTED)
    ax.set_xlabel("Labeled samples used (500 initial + queried)")
    ax.set_ylabel("F1 (Left)")
    ax.set_xlim(400, 6900)
    ax.legend(loc="lower right", fontsize=9.5)
    finish(fig, "fig2_learning_curves.png")


# ------------------------------------------------- fig 3: dup sweep
def fig_dup():
    df = pd.read_csv(RES / "dup_sweep.csv")
    fig, ax = new_ax(
        "Rebalancing is the single largest lever (+0.05)",
        "Duplicate each positive training row ×ratio → shifts the fixed 0.5 vote threshold · uncertainty-selected 5,500 labels",
    )
    for seed, g in df.groupby("seed"):
        ax.plot(g["dup"], g["f1"], color=BASELINE, lw=1.3, zorder=2,
                label="individual seeds" if seed == 1 else None)
    m = df.groupby("dup")["f1"].mean()
    ax.plot(m.index, m.values, color=BLUE, lw=2.4, solid_capstyle="round",
            marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="mean of seeds 1–3", zorder=3)
    ax.annotate(f"no rebalance\n{m.loc[0.0]:.3f}", (0.0, m.loc[0.0]),
                xytext=(8, 6), textcoords="offset points", fontsize=9.5, color=INK2)
    ax.annotate(f"broad plateau 1.0–2.5 → ratio chosen per-seed by CV\n(labeled data only, never the test set)",
                xy=(2.0, m.loc[2.0] + 0.001), xytext=(0.62, 0.648),
                fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    ax.set_xlabel("Extra copies per positive row (duplication ratio)")
    ax.set_ylabel("F1 (Left)")
    ax.legend(loc="lower right", fontsize=9.5)
    finish(fig, "fig3_dup_sweep.png")


# ------------------------------------------------- fig 4: everything we tried
def fig_tried():
    fb = pd.read_csv(RES / "final_bars.csv")
    dup = pd.read_csv(RES / "dup_sweep.csv")
    v2 = pd.read_csv(RES / "variants2.csv")
    bs = pd.read_csv(RES / "batch_size.csv")
    ps = pd.read_csv(RES / "pseudo.csv")
    nf = pd.read_csv(RES / "noise_filter.csv")
    ct = pd.read_csv(RES / "center.csv")
    h2h = pd.read_csv(RES / "head_to_head.csv")
    cvd = pd.read_csv(RES / "cv_dup.csv")

    def mean_of(df, col, val, target="f1"):
        return df[df[col] == val][target].mean()

    base_v2 = mean_of(v2, "variant", "baseline")
    center_best = ct.groupby(["seed", "center"])["f1"].max().reset_index()
    items = [
        ("Rebalance: duplicate positives",  mean_of(dup, "dup", 2.0) - mean_of(dup, "dup", 0.0)),
        ("Uncertainty sampling (vs random)", mean_of(fb, "variant", "uncertainty 5000")
                                             - mean_of(fb, "variant", "random 5000")),
        ("Repeated-CV ratio pick",          mean_of(h2h, "cfg", "B_unc_repcv")
                                             - cvd.loc[cvd.groupby("seed")["cv_f1"].idxmax(), "test_f1"].mean()),
        ("Diversity-clustered batches",     mean_of(h2h, "cfg", "C_div_repcv")
                                             - mean_of(h2h, "cfg", "B_unc_repcv")),
        ("Pseudo-labeling (t=0.85)",        ps[ps["thresh"] == 0.85]["f1"].mean()
                                             - ps[ps["variant"] == "no_pseudo"]["f1"].mean()),
        ("Label-noise filtering",           nf[nf["filter"] == "t0.90"]["f1"].mean()
                                             - nf[nf["filter"] == "no_filter"]["f1"].mean()),
        ("Batch 250 (vs 500)",              mean_of(bs, "batch", 250) - mean_of(bs, "batch", 500)),
        ("Batch 1000 (vs 500)",             mean_of(bs, "batch", 1000) - mean_of(bs, "batch", 500)),
        ("Harvest likely-positives (20%)",  mean_of(v2, "variant", "pos_harvest_0.2") - base_v2),
        ("Select with rebalanced model",    mean_of(v2, "variant", "loop_dup_1.0") - base_v2),
        ("Query near t=0.35 (not 0.5)",     mean_of(center_best, "center", 0.35)
                                             - mean_of(center_best, "center", 0.5)),
        ("Query near t=0.30 (not 0.5)",     mean_of(center_best, "center", 0.30)
                                             - mean_of(center_best, "center", 0.5)),
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items])

    fig, ax = new_ax(
        "Everything we tried, in one picture",
        "Change in mean F1(Left) over seeds 1–3 when adding each idea to the pipeline · blue helped, red hurt",
    )
    fig.set_size_inches(8.4, 5.2)
    fig.subplots_adjust(left=0.315, top=0.845, bottom=0.11)
    ys = np.arange(len(items))[::-1]
    colors = [BLUE if v > 0.0005 else (RED if v < -0.0005 else BASELINE) for v in vals]
    ax.barh(ys, vals, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=4)
    for y, v in zip(ys, vals):
        ax.text(v + (0.0022 if v >= 0 else -0.0022), y,
                f"{v:+.3f}" if abs(v) >= 0.0005 else "±0.000",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ mean F1 (Left)")
    ax.set_xlim(-0.115, 0.075)
    finish(fig, "fig4_what_we_tried.png")


# ------------------------------------------------- fig 5: cumulative gains
def fig_cumulative():
    fb = pd.read_csv(RES / "final_bars.csv")
    stages = [
        ("Initial 500 labels only", fb[fb["variant"] == "initial 500 only"]["f1"].mean()),
        ("+ spend full budget (random)", fb[fb["variant"] == "random 5000"]["f1"].mean()),
        ("+ uncertainty sampling", fb[fb["variant"] == "uncertainty 5000"]["f1"].mean()),
        ("+ rebalancing (dup ×2.0)", 0.6366),
        ("+ diversity + CV-tuned ratio", fb[fb["variant"] == "diverse uncertainty + rebalance (final)"]["f1"].mean()),
    ]
    fig, ax = new_ax(
        "From skeleton to final: +0.24 F1",
        "Mean F1(Left) over seeds 1–3 as each component is added · ceiling 0.657 = all 14,900 labels + optimal threshold",
    )
    fig.subplots_adjust(left=0.29)
    ys = np.arange(len(stages))[::-1]
    vals = [v for _, v in stages]
    ax.barh(ys, vals, height=0.58, color=RAMP5, zorder=3)
    for y, v, (label, _) in zip(ys, vals, stages):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=10, color=INK2)
    ax.axvline(0.657, color=BASELINE, lw=1.2, zorder=2)
    ax.text(0.6555, len(stages) - 0.45, "noise ceiling 0.657 ", ha="right",
            fontsize=9, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([s for s, _ in stages], fontsize=10.5, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Mean F1 (Left), local test")
    finish(fig, "fig5_cumulative_gains.png")


# ------------------------------------------------- fig 6: CV adaptivity
def fig_cv():
    df = pd.read_csv(RES / "cvgrid.csv")
    m = df.groupby("dup")[["cv_f1", "test_f1"]].mean()
    fig, ax = new_ax(
        "The CV estimate tracks the hidden objective",
        "Duplication-ratio choice: repeated 3-fold CV on labeled data only vs actual local-test F1 · mean of seeds 1–3",
    )
    ax.axvspan(2.0, 2.5, color=GRID, alpha=0.45, zorder=1)
    ax.plot(m.index, m["test_f1"], color=BLUE, lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            label="local test F1 (never seen by strategy)")
    ax.plot(m.index, m["cv_f1"], color=GREEN, lw=2, marker="o", ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            label="CV estimate (labeled data only)")
    ax.text(2.25, float(m["test_f1"].max()) + 0.0012, "per-seed CV picks\n(2.0 / 2.25 / 2.5)",
            ha="center", fontsize=9.5, color=INK2)
    ax.set_xlabel("Duplication ratio")
    ax.set_ylabel("F1 (Left)")
    ax.set_ylim(float(m["cv_f1"].min()) - 0.003, float(m["test_f1"].max()) + 0.006)
    ax.legend(loc="lower right", fontsize=9.5)
    finish(fig, "fig6_cv_adaptivity.png")


# ------------------------------------------------- fig 7: final per-seed
def fig_final():
    df = pd.read_csv(RES / "final_bars.csv")
    order = ["initial 500 only", "random 5000", "uncertainty 5000",
             "diverse uncertainty + rebalance (final)"]
    names = ["initial 500 only", "random full budget", "uncertainty sampling",
             "final: + diversity + rebalance"]
    fig, ax = new_ax(
        "Final per-seed results — mean F1 = 0.644",
        "F1(Left) on the local test sets · every stage improves every seed",
    )
    fig.subplots_adjust(top=0.76)
    seeds = [1, 2, 3]
    xs = np.arange(len(seeds))
    w = 0.19
    for i, (v, name) in enumerate(zip(order, names)):
        g = df[df["variant"] == v].set_index("seed").loc[seeds, "f1"]
        bars = ax.bar(xs + (i - 1.5) * (w + 0.015), g.values, w, color=RAMP4[i],
                      label=name, zorder=3)
        if i == len(order) - 1:
            for b, val in zip(bars, g.values):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.008, f"{val:.3f}",
                        ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=11, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("F1 (Left)")
    ax.set_ylim(0, 0.72)
    fig.legend(loc="upper left", bbox_to_anchor=(0.055, 0.885), ncol=4,
               fontsize=8.6, frameon=False, columnspacing=1.1, handlelength=1.4)
    finish(fig, "fig7_final_bars.png")


if __name__ == "__main__":
    fig_diagnosis()
    fig_curves()
    fig_dup()
    fig_tried()
    fig_cumulative()
    fig_cv()
    fig_final()
