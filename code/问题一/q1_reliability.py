"""Compute the eight-airport Q1 statistics and generate the reliability figure."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bts_batch import AIRPORTS
from bts_monthly import OUTCOMES, THRESHOLDS, integer, sha256


ROOT = Path(__file__).resolve().parents[2]
POOLED = "ALL_EIGHT"
TIMES = np.arange(181)
EVENTS = ("ever_diverted", "final_reached", "final_late_60", *(f"severe_{t}" for t in THRESHOLDS))
ANCHORS = (("undiverted_on_time", 15, True), *((f"severe_{t}", t, False) for t in THRESHOLDS))
COLUMNS = ["Origin", "is_selected_origin", "outcome_code", "ArrDelay", "ArrDelayMinutes",
           "DivArrDelay", "undiverted_on_time", *EVENTS]
COUNT_FIELDS = ["n", *(f"n_{state}" for state in OUTCOMES),
                *(name for event in EVENTS for name in (f"n_{event}", f"n_{event}_unknown"))]


def selected_months(directories, years):
    expected = {f"{year}-{month:02d}" for year in years for month in range(1, 13)}
    located = {}
    for directory in directories:
        manifest_path = directory / "manifest.json"
        batch = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(batch["airports"]) != set(AIRPORTS):
            raise ValueError(f"Unexpected airport coverage in {manifest_path}")
        for month in expected.intersection(batch["months"]):
            if month in located:
                raise ValueError(f"Month {month} was supplied more than once")
            located[month] = directory / month
    if missing := expected.difference(located):
        raise ValueError("Missing audited months: " + ", ".join(sorted(missing)))
    return sorted(located.items())


def set_exact(lower, upper, eligible, values):
    known = eligible & np.isfinite(values)
    lower[known] = values[known]
    upper[known] = values[known]


def timing_bounds(frame, direct, signed, positive, diverted_delay):
    """Unknown delays are intervals. Nonarrival has delay +infinity for this event."""
    state = frame.outcome_code.to_numpy()
    reached = frame.final_reached.eq("True").to_numpy()
    not_reached = frame.final_reached.eq("False").to_numpy()
    ever_diverted = frame.ever_diverted.eq("True").to_numpy()
    diversion = state == "V"
    cancelled = state == "C"
    definite_direct = direct | (reached & frame.ever_diverted.eq("False").to_numpy() & ~cancelled)
    low = np.full(len(frame), -np.inf)
    high = np.full(len(frame), np.inf)
    low[cancelled | diversion | not_reached | ever_diverted] = np.inf

    # A clipped zero implies arrival delay of at most zero. It does not establish an early arrival.
    clipped_only = definite_direct & ~np.isfinite(signed) & (positive >= 0)
    high[clipped_only & (positive == 0)] = 0
    set_exact(low, high, clipped_only & (positive > 0), positive)
    set_exact(low, high, definite_direct, signed)

    for flag, threshold, yes_means_success in ANCHORS:
        successful = frame[flag].eq("True" if yes_means_success else "False").to_numpy()
        unsuccessful = frame[flag].eq("False" if yes_means_success else "True").to_numpy()
        high[successful] = np.minimum(high[successful], threshold - 1)
        low[unsuccessful] = np.maximum(low[unsuccessful], threshold)

    wide_low = low.copy()
    wide_high = high.copy()
    potentially_diverted_arrival = (diversion | ever_diverted) & ~not_reached & ~cancelled
    wide_low[potentially_diverted_arrival] = -np.inf
    wide_high[potentially_diverted_arrival] = np.inf
    set_exact(wide_low, wide_high, potentially_diverted_arrival & reached, diverted_delay)

    # An unresolved outcome can allow a diverted arrival even if strict on-time failed.
    unresolved_diversion = ((state == "U") & ~not_reached & ~potentially_diverted_arrival
                            & ~frame.ever_diverted.eq("False").to_numpy())
    wide_low[unresolved_diversion] = -np.inf
    late = frame.final_late_60.eq("True").to_numpy()
    reached_with_no_long_delay = reached & frame.final_late_60.eq("False").to_numpy()
    wide_low[late] = np.maximum(wide_low[late], 60)
    wide_high[reached_with_no_long_delay] = np.minimum(wide_high[reached_with_no_long_delay], 59)

    if (low > high).any() or (wide_low > wide_high).any():
        raise ValueError("Conflicting evidence generated an empty delay interval")
    return low, high, wide_low, wide_high


def period_counts(group, month, airport):
    direct = group.outcome_code.isin(["T", "M", "L"]).to_numpy()
    signed = integer(group.ArrDelay).to_numpy(dtype=float, na_value=np.nan)
    positive = integer(group.ArrDelayMinutes).to_numpy(dtype=float, na_value=np.nan)
    div_delay = integer(group.DivArrDelay).to_numpy(dtype=float, na_value=np.nan)
    signed_sample = signed[direct & np.isfinite(signed)]
    reached_after_diversion = group.outcome_code.eq("V") & group.final_reached.eq("True")
    reached_without_diversion = group.final_reached.eq("True") & group.ever_diverted.eq("False")
    counters = {"month": month, "airport": airport, "n": len(group),
                **{f"n_{s}": int(group.outcome_code.eq(s).sum()) for s in OUTCOMES},
                "n_ever_diverted_and_cancelled": int((group.outcome_code.eq("C") & group.ever_diverted.eq("True")).sum()),
                "n_v_reached": int(reached_after_diversion.sum()),
                "n_strict_final_yes": int((reached_without_diversion | direct).sum()),
                "n_v_reached_without_delay": int((reached_after_diversion.to_numpy() & ~np.isfinite(div_delay)).sum()),
                "n_valid_signed": len(signed_sample),
                "signed_delay_sum": int(signed_sample.sum()),
                "nonnegative_delay_sum": int(np.maximum(signed_sample, 0).sum()),
                "n_early": int((signed_sample < 0).sum())}
    for event in EVENTS:
        counters[f"n_{event}"] = int(group[event].eq("True").sum())
        counters[f"n_{event}_unknown"] = int(group[event].eq("").sum())
    low, high, wide_low, wide_high = timing_bounds(group, direct, signed, positive, div_delay)
    strict_yes = np.searchsorted(np.sort(high), TIMES, side="left")
    strict_possible = np.searchsorted(np.sort(low), TIMES, side="left")
    wide_yes = np.searchsorted(np.sort(wide_high), TIMES, side="left")
    wide_possible = np.searchsorted(np.sort(wide_low), TIMES, side="left")
    curve = np.column_stack((strict_yes, strict_possible, wide_yes, wide_possible))
    validate_curve(counters, curve, month, airport)
    return counters, curve


def validate_curve(counts, curve, month, airport):
    n = counts["n"]
    if (np.diff(curve, axis=0) < 0).any() or (curve[:, [0, 2]] > curve[:, [1, 3]]).any():
        raise ValueError(f"Invalid reliability bounds for {month} {airport}")
    if (curve[:, :2] > curve[:, 2:]).any() or (curve > n).any():
        raise ValueError(f"Invalid strict versus inclusive counts for {month} {airport}")
    if curve[15, 0] != counts["n_T"]:
        raise ValueError(f"The 15-minute anchor differs for {month} {airport}")
    for threshold in THRESHOLDS:
        lower = counts[f"n_severe_{threshold}"]
        upper = lower + counts[f"n_severe_{threshold}_unknown"]
        if tuple(n - curve[threshold, [1, 0]]) != (lower, upper):
            raise ValueError(f"Severe-disruption complement differs for {month} {airport} at {threshold} minutes")


def add_metrics(totals):
    n = totals.n
    result = totals.copy()
    result["n_A"] = totals[["n_T", "n_M", "n_L"]].sum(axis=1)
    result["R1"] = totals.n_T / n
    result["R1_conservative_upper"] = (totals.n_T + totals.n_U) / n
    result["R2"] = totals.n_T / (n - totals.n_U).replace(0, np.nan)
    result["R3"] = totals.n_T / result.n_A.replace(0, np.nan)
    result["denominator_gap_pp"] = 100 * (result.R3 - result.R1)
    expected_gap = 100 * result.R3 * (totals.n_C + totals.n_V + totals.n_U) / n
    np.testing.assert_allclose(result.denominator_gap_pp, expected_gap, atol=1e-12, rtol=0)
    available = result.R3.notna() & result.R2.notna()
    assert result.loc[available, "R3"].ge(result.loc[available, "R2"]).all()
    assert result.loc[available, "R2"].ge(result.loc[available, "R1"]).all()
    for state in OUTCOMES:
        result[f"rate_{state}"] = totals[f"n_{state}"] / n
    for event in EVENTS:
        result[f"{event}_lower"] = totals[f"n_{event}"] / n
        result[f"{event}_upper"] = (totals[f"n_{event}"] + totals[f"n_{event}_unknown"]) / n
    result["R4"] = result.final_reached_lower
    result["mean_signed_minutes"] = totals.signed_delay_sum / totals.n_valid_signed.replace(0, np.nan)
    result["mean_nonnegative_minutes"] = totals.nonnegative_delay_sum / totals.n_valid_signed.replace(0, np.nan)
    return result


def curve_tables(metrics, curves):
    frames = []
    for airport, matrix in curves.items():
        n = metrics.loc[airport, "n"]
        data = pd.DataFrame(matrix, columns=["strict_yes", "strict_possible", "wide_yes", "wide_possible"])
        data.insert(0, "threshold_minutes", TIMES)
        data.insert(0, "airport", airport)
        data["n"] = n
        for label in ["strict", "wide"]:
            data[f"{label}_lower"] = data[f"{label}_yes"] / n
            data[f"{label}_upper"] = data[f"{label}_possible"] / n
            data[f"{label}_unknown"] = data[f"{label}_possible"] - data[f"{label}_yes"]
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def comparisons(metrics, curve):
    clean = metrics.drop(index=POOLED)
    results = []

    def add(label, base, variant, high_is_better=True):
        frame = pd.DataFrame({"base_rate": base, "variant_rate": variant})
        frame["base_rank"] = base.rank(ascending=not high_is_better, method="min").astype("Int64")
        frame["variant_rank"] = variant.rank(ascending=not high_is_better, method="min").astype("Int64")
        frame["delta_pp"] = 100 * (variant - base)
        frame["rank_change"] = frame.base_rank - frame.variant_rank
        frame.insert(0, "comparison", label)
        results.append(frame.reset_index())

    add("R3_to_R1", clean.R3, clean.R1)
    add("R2_to_R1", clean.R2, clean.R1)
    for threshold in (45, 90):
        add(f"D60_to_D{threshold}", clean.severe_60_lower, clean[f"severe_{threshold}_lower"], False)
    for threshold in (15, 45, 60, 90, 180):
        points = curve.loc[curve.threshold_minutes.eq(threshold)].set_index("airport").loc[clean.index]
        add(f"strict_to_wide_{threshold}", points.strict_lower, points.wide_lower)
    at_15 = curve.loc[curve.threshold_minutes.eq(15)].set_index("airport").loc[clean.index]
    at_16 = curve.loc[curve.threshold_minutes.eq(16)].set_index("airport").loc[clean.index]
    add("strict_15_to_16", at_15.strict_lower, at_16.strict_lower)
    add("strict_to_wide_final", clean.n_strict_final_yes / clean.n, clean.R4)
    return pd.concat(results, ignore_index=True)


def make_figure(curve, figure_folder, years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = ("#4573B4", "#D73221", "#FCB777", "#6491C1", "#E35235", "#FDE699", "#E48070", "#D2EDF2")
    patterns = ("-", "--", "-.", (0, (2, 2)), (0, (6, 2, 1, 2)), (0, (4, 2)),
                (0, (1, 1)), (0, (8, 2, 2, 2)))
    fig, ax = plt.subplots(figsize=(8.1, 4.6), layout="constrained")
    for airport, color, pattern in zip(AIRPORTS, colors, patterns):
        points = curve.loc[curve.airport.eq(airport)]
        # With integer delays, the strict event d<t jumps immediately to the right of each minute.
        line = ax.step(points.threshold_minutes, points.strict_lower, where="pre",
                       label=airport, linewidth=1.65, color=color, linestyle=pattern, zorder=3)[0]
        if color in {"#FCB777", "#FDE699", "#D2EDF2"}:
            line.set_path_effects([effects.Stroke(linewidth=2.2, foreground="#515b66"), effects.Normal()])
    for threshold in (15, *THRESHOLDS):
        ax.axvline(threshold, color="#4573B4" if threshold == 15 else "#a2adb8",
                   alpha=0.85, lw=1.25 if threshold == 15 else 0.75, ls=":" if threshold == 15 else "--",
                   zorder=1)
    first_point = curve.loc[curve.threshold_minutes.eq(0), "strict_lower"].min()
    lower_limit = max(0, np.floor(first_point * 20) / 20)
    ax.set(xlim=(0, 180), ylim=(lower_limit, 1.008),
           xticks=(0, 15, 30, 45, 60, 90, 120, 150, 180),
           xlabel="Maximum arrival delay (minutes, exclusive)",
           ylabel="Share of scheduled departures confirmed on target")
    ax.set_yticks(np.arange(np.ceil(lower_limit * 10) / 10, 1.01, 0.1).round(2))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#d8dfe5", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncols=2, frameon=False, handlelength=2.6, fontsize=9)
    ax.margins(x=0)
    fig.savefig(figure_folder / "reliability_curves.pdf",
                metadata={"Title": f"Strict flight reliability, {years[0]}–{years[-1]}"})
    fig.savefig(figure_folder / "reliability_curves.png", dpi=240)
    plt.close(fig)
    return matplotlib.__version__


def run(args):
    paths = selected_months(args.input_dir or [ROOT / "data/analysis/development"], args.years)
    monthly = []
    curve_totals = {airport: np.zeros((len(TIMES), 4), dtype=np.int64) for airport in AIRPORTS}
    inputs = []
    for month, folder in paths:
        path = folder / "flights.csv.gz"
        frames = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
        selected = frames.loc[frames.is_selected_origin.eq("True")]
        if set(selected.Origin) != set(AIRPORTS):
            raise ValueError(f"Incomplete eight-airport sample in {path}")
        audit_path = folder / "airport_summary.csv"
        audited = pd.read_csv(audit_path).set_index("origin")
        for airport, group in selected.groupby("Origin"):
            counts, curve = period_counts(group, month, airport)
            if not all(counts[column] == audited.loc[airport, column] for column in COUNT_FIELDS):
                raise ValueError(f"Recount disagrees with {audit_path}: {airport}")
            monthly.append(counts)
            curve_totals[airport] += curve
        provenance = folder / "manifest.json"
        inputs.append({"month": month, "flight_file": str(path.resolve()), "sha256": sha256(path),
                       "summary_sha256": sha256(audit_path), "audit_manifest_sha256": sha256(provenance)})
        print(f"{month}: matched monthly audit for {len(selected):,} scheduled departures", flush=True)

    month_counts = pd.DataFrame(monthly)
    totals = month_counts.groupby("airport").sum(numeric_only=True)
    totals.loc[POOLED] = totals.sum()
    metrics = add_metrics(totals)
    curve_totals[POOLED] = sum(curve_totals.values())
    for airport, matrix in curve_totals.items():
        validate_curve(totals.loc[airport], matrix, "all requested months", airport)
    curves = curve_tables(metrics, curve_totals)
    definitions = comparisons(metrics, curves)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as out, TemporaryDirectory(dir=args.figures) as fig:
        output, figure = Path(out), Path(fig)
        metrics.reset_index().to_csv(output / "airport_metrics.csv", index=False)
        add_metrics(month_counts.set_index(["month", "airport"])).reset_index().to_csv(
            output / "monthly_metrics.csv", index=False)
        curves.to_csv(output / "reliability_curves.csv", index=False)
        definitions.to_csv(output / "definition_comparison.csv", index=False)
        matplotlib_version = make_figure(curves, figure, args.years)
        manifest = {
            "years": args.years, "months": [month for month, _ in paths], "airports": list(AIRPORTS),
            "outcome_threshold_minutes": 60, "on_time_threshold_minutes": 15,
            "sensitivity_threshold_minutes": list(THRESHOLDS), "curve_max_minutes": 180,
            "planned_departures": int(totals.loc[POOLED, "n"]),
            "validation": {"month_airport_audits_matched": len(monthly), "pooled_curve_validated": True,
                           "strict_15_and_severe_complement_validated": True},
            "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib_version, "argv": [sys.executable, *sys.argv],
            "script_sha256": sha256(Path(__file__)), "inputs": inputs,
            "outputs": {file.name: sha256(file) for file in [*output.iterdir(), *figure.iterdir()]},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for path in figure.iterdir():
            path.replace(args.figures / path.name)
        for path in sorted(output.iterdir(), key=lambda item: item.name == "manifest.json"):
            path.replace(args.output / path.name)
    print(f"Validated {len(monthly)} airport-months and {int(totals.loc[POOLED, 'n']):,} departures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--input-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/问题一计算结果")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures/问题一计算结果")
    args = parser.parse_args()
    args.years = sorted(set(args.years))
    run(args)


if __name__ == "__main__":
    main()
