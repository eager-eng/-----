"""Render Q3 report figures from frozen evaluation summaries."""

from q3_figures import BLUE, ORANGE, RED, plt, save
from q3_inputs import ROOT
import pandas as pd


EVENTS = {"D60": "Severe disruption", "C": "Cancellation", "L60": "Un-diverted arrival ≥60 min late"}
MODELS = {
    "B0": (ORANGE, "B0 · Historical frequencies"),
    "A": (BLUE, "Info-A · Schedule and history"),
    "B": (RED, "Info-B · Adds archived forecast air temperature"),
}
WINDOWS = {2025: "2025 · Jan–Dec", 2026: "2026 · Jan–Sep"}


def calibration_figure(period, directory, year):
    curves = pd.read_csv(period / "calibration.csv")
    parts = pd.read_csv(period / "brier_decomposition.csv").set_index("model").loc[list(MODELS)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout="constrained")
    for ax, (event, title) in zip(axes, EVENTS.items()):
        panel = curves.loc[curves.event.eq(event) & curves.model.isin(MODELS)]
        edge = 105 * panel[["observed", "predicted"]].max().max()
        ax.plot([0, edge], [0, edge], color=".65", lw=.8)
        for model, (color, label) in MODELS.items():
            line = panel.loc[panel.model.eq(model)].sort_values("bin")
            ax.plot(100 * line.predicted, 100 * line.observed, "o-", lw=1.3, ms=3,
                    color=color, label=label)
        ax.set(xlim=(0, edge), ylim=(0, edge), xlabel="Predicted risk (%)", title=title)
    axes[0].set_ylabel("Observed event rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle(f"{WINDOWS[year]} | Reliability after frozen temperature scaling for Info-A and Info-B")

    ax = axes[3]
    for index, (model, row) in enumerate(parts.iterrows()):
        ax.bar([index - .18, index + .18], 1000 * row[["reliability", "resolution"]],
               width=.32, color=MODELS[model][0], edgecolor=".4", lw=.4)
    ax.set_xticks(range(len(MODELS)), ["B0", "Info-A", "Info-B"])
    ax.set(title="Severe disruption · Brier components", ylabel="Component × 1,000")
    ax.text(.02, .98, "Left: reliability\nRight: resolution", transform=ax.transAxes,
            va="top", fontsize=8)
    save(fig, directory, f"q3_calibration_{year}")


def information_figure(period, directory, year):
    scores = pd.read_csv(period / "metrics.csv")
    contrasts = pd.read_csv(period / "paired_brier_intervals.csv")
    paired = contrasts.loc[contrasts["first"].eq("B") & contrasts["second"].eq("A")]
    events = list(EVENTS)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.3, 3.4), layout="constrained")
    for scope, delta, color, label in (("all", -.12, RED, "All scored flights"),
                                       ("both_covered", .12, ORANGE, "Forecast at both ends")):
        rows = paired.loc[paired.scope.eq(scope)].set_index("event").loc[events]
        y = [index + delta for index in range(len(events))]
        ax.hlines(y, 1000 * rows.lo, 1000 * rows.hi, color=color, lw=1)
        ax.scatter(1000 * rows.brier_difference, y, c=color, s=19, label=label)
        wide = scores.loc[scores.scope.eq(scope) & scores.model.isin(["A", "B"])].pivot(
            index="event", columns="model", values="ap"
        )
        bx.scatter(100 * (wide.loc[events, "B"] - wide.loc[events, "A"]), y, c=color, s=19)
    for axis, label in ((ax, "Brier difference × 1,000"), (bx, "Average precision difference (pp)")):
        axis.axvline(0, color=".55", lw=.8)
        axis.set_yticks(range(len(events)), events)
        axis.invert_yaxis()
        axis.set_xlabel(label)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(f"{WINDOWS[year]} | Adding archived forecast air temperature (Info-B − Info-A)")
    save(fig, directory, f"q3_information_gain_{year}")


if __name__ == "__main__":
    directory = ROOT / "figures" / "问题三计算结果"
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        for year in WINDOWS:
            period = ROOT / "outputs" / "问题三计算结果" / str(year)
            calibration_figure(period, directory, year)
            information_figure(period, directory, year)
