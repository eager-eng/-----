"""Render paper figures from the Q2 result tables."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, ORANGE, PALE = "#4573B4", "#D73221", "#FCB777", "#D2EDF2"
METRICS = ("R1", "D60")
SCENARIOS = ("full", "time_distance", "common_carrier", "matched_route_carrier_time")
LABELS = ("All scheduled", "Time + distance", "Shared carriers", "Matched route, carrier, time")


def finish(fig, directory, name):
    files = [directory / f"{name}.{ext}" for ext in ("pdf", "png")]
    for path in files:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return files


def pair_figure(pairs, directory):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 7.0), layout="constrained")
    for axis, metric in zip(axes, METRICS):
        rows = pairs.loc[pairs.metric.eq(metric)].sort_values("adjusted_gap_pp")
        ypos = np.arange(len(rows))
        axis.axvspan(-1, 1, color=PALE, alpha=.65)
        axis.axvline(0, color=".45", linewidth=.8)
        for stem, color, offset, label in (("crude", ORANGE, -.17, "Crude"),
                                            ("adjusted", BLUE, .17, "Standardized")):
            point = rows[f"{stem}_gap_pp"].to_numpy()
            axis.errorbar(point, ypos+offset, xerr=[point-rows[f"{stem}_lo_pp"], rows[f"{stem}_hi_pp"]-point],
                          fmt="o", color=color, markersize=3, elinewidth=1, label=label)
        axis.set_yticks(ypos, rows.a + " − " + rows.b)
        axis.set_xlabel(f"{metric} advantage of first airport (pp)")
        axis.grid(axis="y", alpha=.16)
        axis.legend(loc="lower right", frameon=False, fontsize=8)
    return finish(fig, directory, "rank_adjustment_intervals")


def seasonal_figure(results, directory):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.1), layout="constrained")
    season = results["four_dimensions"].query("dimension == 'season'")
    time = results["four_dimensions"].query("dimension == 'time'")
    pattern = results["sfo_atl_inference"]
    profile = results["sfo_atl_profiles"]
    for axis, table, order, ticklabels, metric, xlabel in (
        (axes[0, 0], season, ["DJF","MAM","JJA","SON"], ["Winter","Spring","Summer","Fall"], "R1", "Season"),
        (axes[0, 1], time, [0,1,2,3], ["00–09","09–14","14–18","18–24"], "D60", "Scheduled departure hour"),
    ):
        for airport, row in table.groupby("airport"):
            values = row.set_index("level")[metric].reindex(order).to_numpy()
            color = BLUE if airport == "SFO" else RED if airport == "ATL" else ".76"
            strong = airport in ("SFO", "ATL")
            axis.plot(range(4), 100*values, "o-", color=color, linewidth=2 if strong else .8,
                      markersize=4 if strong else 2, label=airport if strong else None, zorder=3 if strong else 2)
        axis.set_xticks(range(4), ticklabels)
        axis.set(xlabel=xlabel, ylabel=f"{metric} (%)")
        axis.legend(frameon=False)
    axis = axes[1, 0]
    y = np.arange(len(SCENARIOS))
    view = pattern.loc[pattern.year.eq("pooled") & pattern.metric.eq("R1")].set_index("scenario").loc[list(SCENARIOS)]
    axis.axvline(0, color=".45", linewidth=.8)
    for season_name, color, shift in (("winter", BLUE, -.13), ("summer", RED, .13)):
        point = view[f"{season_name}_gap_pp"]
        axis.errorbar(point, y+shift, xerr=[point-view[f"{season_name}_lo_pp"],view[f"{season_name}_hi_pp"]-point],
                      fmt="o", color=color, label=season_name.title(), markersize=4, elinewidth=1.1)
    axis.set_yticks(y, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("SFO − ATL on-time rate (pp)")
    axis.legend(loc="lower left", frameon=False, fontsize=8)
    axis = axes[1, 1]
    view = (profile.loc[profile.scenario.eq("full") & profile.year.eq("pooled")]
            .set_index(["airport", "season"]).loc[[("ATL","DJF"),("ATL","JJA"),("SFO","DJF"),("SFO","JJA")]])
    previous = np.zeros(len(view))
    for field, label, color in (("R_L60","Arrived ≥60 min late",ORANGE),("RC","Canceled",RED),
                                ("RV","Diverted","#E48070")):
        values = 100*view[field].to_numpy()
        axis.bar(range(len(view)), values, bottom=previous, color=color, label=label)
        previous += values
    axis.set_xticks(range(4), ["ATL winter","ATL summer","SFO winter","SFO summer"], rotation=12)
    axis.set_ylabel("Severe disruptions (%)")
    axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.spines[["top","right"]].set_visible(False)
    return finish(fig, directory, "seasonal_variation_and_reversal")


def render(results, primary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        return pair_figure(primary, directory) + seasonal_figure(results, directory)
