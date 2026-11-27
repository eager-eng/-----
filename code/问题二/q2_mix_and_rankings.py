"""Compute Q2 flight-mix, rank and seasonal reversal estimates from audited months."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import sys

# These matrix products are small enough that many BLAS threads add overhead.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "问题一"))
from q1_reliability import AIRPORTS, selected_months, sha256

AIRPORTS = tuple(sorted(AIRPORTS))
PAIR = ("ATL", "SFO")
SEASONS = ("DJF", "MAM", "JJA", "SON")
METRICS = {"R1": "T", "D60": "D"}
COUNTS = ["N", "T", "D", "C", "V", "L60", "U", "D_unknown", "quality",
          "conditional_missing"]
STRATA = ["season", "time", "band"]
KEYS = ["date", "airport", *STRATA, "dow", "carrier", "dest"]
QUALITY = ["quality_status_conflict", "quality_delay_conflict",
           "quality_diversion_reach_conflict", "quality_invalid_field",
           "quality_missing_final_div_delay"]
FIELDS = ["FlightDate", "Origin", "DestAirportID", "Reporting_Airline", "Distance",
          "DayOfWeek", "CRSDepTime_hhmm", "CRSDepTime_day_offset", "is_selected_origin",
          "is_selected_destination", "outcome_code", "severe_60", "ArrDelay",
          "CancellationCode", *QUALITY]


def calendar_season(month):
    return np.array(SEASONS)[(np.asarray(month) % 12) // 3]


def read_counts(located):
    daily_counts = []
    sources = []
    scopes = np.zeros(3, dtype=np.int64)
    for month, directory in located:
        source = directory / "flights.csv.gz"
        with_path = pd.read_csv(source, usecols=FIELDS, dtype=str, keep_default_na=False)
        out = with_path.is_selected_origin.eq("True")
        arriving = with_path.is_selected_destination.eq("True")
        scopes += [(out & arriving).sum(), (out & ~arriving).sum(), (~out & arriving).sum()]
        frame = with_path.loc[out].copy()
        frame["date"] = pd.to_datetime(frame.FlightDate, format="%Y-%m-%d")
        frame["season"] = calendar_season(frame.date.dt.month)
        clock = pd.to_numeric(frame.CRSDepTime_hhmm, errors="coerce")
        offset = pd.to_numeric(frame.CRSDepTime_day_offset, errors="coerce")
        if not ((clock.between(0, 2359)) & (clock % 100 < 60)
                & (offset.eq(0) | (offset.eq(1) & clock.eq(0)))).all():
            raise ValueError(f"Invalid normalized departure clock in {month}")
        frame["time"] = pd.cut(clock, [-1, 859, 1359, 1759, 2359], labels=False)
        distance = pd.to_numeric(frame.Distance, errors="coerce")
        frame["band"] = pd.cut(distance, [-1, 499, 1499, np.inf], labels=False)
        frame = frame.rename(columns={"Origin": "airport", "DestAirportID": "dest",
                                      "Reporting_Airline": "carrier", "DayOfWeek": "dow"})
        if frame[KEYS].isna().any().any() or frame[KEYS].eq("").any().any():
            raise ValueError(f"Missing Q2 grouping fields in {month}")
        if not frame.outcome_code.isin(["C", "V", "L", "M", "T", "U"]).all():
            raise ValueError(f"Unexpected Q1 outcome in {month}")
        frame["N"] = 1
        for marker in ("T", "C", "V", "U"):
            frame[marker] = frame.outcome_code.eq(marker)
        frame["D"] = frame.severe_60.eq("True")
        frame["D_unknown"] = ~frame.severe_60.isin(["True", "False"])
        delay = pd.to_numeric(frame.ArrDelay, errors="coerce")
        direct = frame.outcome_code.isin(["T", "M", "L"])
        frame["L60"] = direct & delay.ge(60)
        frame["quality"] = frame[QUALITY].eq("True").any(axis=1) | frame["U"]
        frame["conditional_missing"] = (direct & delay.isna()) | (
            frame["C"] & ~frame.CancellationCode.isin(["A", "B", "C", "D"]))
        if not frame.D.eq(frame.C | frame.V | frame.L60).all():
            raise ValueError(f"Q1 severe count differs from its audited components in {month}")
        daily_counts.append(frame.groupby(KEYS, observed=True, sort=False)[COUNTS].sum().reset_index())
        sources.append({"month": month, "file": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                        "sha256": sha256(source), "audit_manifest_sha256": sha256(directory / "manifest.json")})
    print(f"Read and checked {len(located)} months", flush=True)
    return pd.concat(daily_counts, ignore_index=True), scopes, sources


def add_rates(frame):
    result = frame.copy()
    for name, events in {**METRICS, "RC": "C", "RV": "V", "R_L60": "L60"}.items():
        result[name] = result[events] / result.N
    return result


def shared(data, strata, airports, minimum=1):
    counts = data.groupby([*strata, "airport"], observed=True).N.sum().unstack(fill_value=0)
    counts = counts.reindex(columns=list(airports), fill_value=0)
    cells = counts.index[(counts >= minimum).all(axis=1)]
    keys = cells.to_frame(index=False)
    return data.loc[data.airport.isin(airports)].merge(keys, on=strata), keys


def array_counts(data, strata, airports, cells, columns):
    index = pd.MultiIndex.from_frame(cells.merge(pd.DataFrame({"airport": airports}), how="cross")
                                    [["airport", *strata]].sort_values(["airport", *strata]))
    # Use airport-major order with the same sorted cell order for each airport.
    table = data.groupby(["airport", *strata], observed=True)[columns].sum().reindex(index)
    counts = table.to_numpy(dtype=float).reshape(len(airports), len(cells), len(columns))
    if not np.isfinite(counts).all() or (counts[..., 0] <= 0).any():
        raise ValueError("Reference includes an unsupported cell")
    return counts


def support_summary(sample, cells, base, label):
    size = sample.groupby("airport", observed=True).N.sum().reindex(base.index, fill_value=0)
    return pd.DataFrame({"scope": label, "airport": base.index, "cells": len(cells),
                         "n": size.to_numpy(), "coverage": (size / base).to_numpy()})


def reference(counts, equal=False):
    n = counts[..., 0]
    if equal:
        return (n / n.sum(axis=-1, keepdims=True)).mean(axis=0)
    return n.sum(axis=0) / n.sum()


def rates(counts, weights=None):
    n = counts[..., 0]
    if weights is None:
        return counts[..., 1:].sum(axis=-2) / n.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.einsum("...akm,k->...am", counts[..., 1:] / n[..., None], weights)


def bootstrap_weights(dates, block, reps, seed):
    if block < 1:
        raise ValueError("Block length must be positive")
    rng = np.random.default_rng(seed)
    seasons = calendar_season(dates.month)
    starts = np.where(np.r_[True, (seasons[1:] != seasons[:-1]) |
                             (dates.year[1:] != dates.year[:-1])])[0]
    weights = np.zeros((reps, len(dates)), dtype=float)
    for start, stop in zip(starts, np.r_[starts[1:], len(dates)]):
        length = stop - start
        size = min(block, length)
        n_blocks = (length + size - 1) // size
        chosen = rng.integers(start, stop - size + 1, size=(reps, n_blocks, 1))
        days = (chosen + np.arange(size)).reshape(reps, -1)[:, :length]
        np.add.at(weights, (np.arange(reps)[:, None], days), 1)
    return weights


def bootstrap_counts(data, strata, airports, cells, dates, draws, columns):
    cells = cells.sort_values(strata)
    keys = ["airport", *strata]
    merged = (pd.DataFrame({"airport": airports}).merge(cells, how="cross")
              .sort_values(keys).reset_index(drop=True))
    merged["cell"] = np.arange(len(merged))
    table = data.merge(merged, on=keys)[["date", "cell", *columns]]
    day = dates.get_indexer(table.date)
    if (day < 0).any():
        raise ValueError("Unmatched service date")
    stats = np.zeros((len(dates), len(merged), len(columns)), dtype=float)
    np.add.at(stats, (day, table.cell.to_numpy()), table[columns].to_numpy(dtype=float))
    return (draws @ stats.reshape(len(dates), -1)).reshape(
        len(draws), len(airports), len(cells), len(columns))


def intervals(points, draws):
    """Centered, bootstrap SD-standardized simultaneous intervals, with max-t cutoff."""
    finite = np.isfinite(draws).all(axis=1)
    clean = draws[finite]
    if len(clean) < 2:
        raise ValueError("Too few complete bootstrap draws")
    scale = clean.std(axis=0, ddof=1)
    if (scale <= 0).any():
        raise ValueError("Zero bootstrap variance in simultaneous family")
    pivots = np.max(np.abs((clean - points) / scale), axis=1)
    critical = np.quantile(pivots, .95)
    return points - critical * scale, points + critical * scale, int(finite.sum())


def main_comparison(data, dates, boot, block, periods, seed):
    airports = AIRPORTS
    ab = list(combinations(range(len(airports)), 2))
    a, b = np.array(ab).T
    columns = ["N", "T", "D"]
    full = data.groupby("airport", observed=True)[COUNTS].sum().reindex(airports)
    main = []
    decompositions = []
    pairs = []
    rankings = []
    bounds = []
    for floor in (1, 50):
        supported, cells = shared(data, STRATA, airports, floor)
        point = array_counts(supported, STRATA, airports, cells, columns)
        point_crude = rates(point)
        n = point[..., 0]
        by_airport = n / n.sum(axis=1, keepdims=True)
        cell_rate = point[..., 1:] / n[..., None]
        for block_length in sorted({1, 3, block}):
            draws = boot if block_length == block else bootstrap_weights(dates, block_length, periods, seed)
            reps = bootstrap_counts(supported, STRATA, airports, cells, dates, draws, columns)
            rep_crude = rates(reps)
            for weighting in ("pooled", "airport_equal"):
                weights = reference(point, equal=weighting == "airport_equal")
                if not np.isclose(weights.sum(), 1):
                    raise ValueError("Reference weights are not normalized")
                point_std = rates(point, weights)
                rep_std = rates(reps, weights)
                for j, metric in enumerate(METRICS):
                    direction = 1 if metric == "R1" else -1
                    crude, adj = direction * point_crude[:, j], direction * point_std[:, j]
                    diff_c, diff_s = crude[a] - crude[b], adj[a] - adj[b]
                    p = np.r_[diff_c, diff_s]
                    zc = direction * (rep_crude[:, a, j] - rep_crude[:, b, j])
                    zs = direction * (rep_std[:, a, j] - rep_std[:, b, j])
                    lo, hi, valid = intervals(p, np.c_[zc, zs])
                    for k, (ai, bi) in enumerate(ab):
                        rev = diff_c[k] * diff_s[k] < 0
                        opposite = ((lo[k] > 0 and hi[k+len(ab)] < 0) or
                                    (hi[k] < 0 and lo[k+len(ab)] > 0))
                        l, h = lo[k+len(ab)], hi[k+len(ab)]
                        pairs.append({
                            "minimum": floor, "weights": weighting, "block": block_length,
                            "metric": metric, "a": airports[ai], "b": airports[bi],
                            "crude_gap_pp": 100*diff_c[k], "crude_lo_pp": 100*lo[k],
                            "crude_hi_pp": 100*hi[k], "adjusted_gap_pp": 100*diff_s[k],
                            "adjusted_lo_pp": 100*l, "adjusted_hi_pp": 100*h,
                            "point_reversal": rev, "supported_reversal": opposite,
                            "unresolved": l <= 0 <= h,
                            **{f"equivalent_{delta:g}pp": l > -delta/100 and h < delta/100
                               for delta in (.5, 1, 1.5)}, "bootstrap_valid": valid,
                        })
                    if block_length != block:
                        continue
                    for scenario, score, lows, highs in (
                        ("common_crude", crude, lo[:len(ab)], hi[:len(ab)]),
                        ("adjusted", adj, lo[len(ab):], hi[len(ab):]),
                    ):
                        better = np.zeros(len(airports), dtype=int)
                        worse = np.zeros(len(airports), dtype=int)
                        for ai, bi, l, h in zip(a, b, lows, highs):
                            if l > 0:
                                better[bi] += 1
                                worse[ai] += 1
                            if h < 0:
                                better[ai] += 1
                                worse[bi] += 1
                        for i, airport in enumerate(airports):
                            rankings.append({
                                "minimum": floor, "weights": weighting, "metric": metric,
                                "scenario": scenario, "airport": airport,
                                "rank": 1 + int((score > score[i]).sum()), "rank_lo": 1 + better[i],
                                "rank_hi": len(airports) - worse[i],
                            })
                if block_length == block:
                    for i, airport in enumerate(airports):
                        counts = full.loc[airport]
                        row = {"airport": airport, "minimum": floor, "cells": len(cells),
                               "weights": weighting, "n": int(counts.N), "common_n": int(n[i].sum()),
                               "coverage": n[i].sum()/counts.N}
                        for j, (metric, event) in enumerate(METRICS.items()):
                            row[f"{metric}_crude"] = counts[event]/counts.N
                            row[f"{metric}_common"] = point_crude[i, j]
                            row[f"{metric}_adjusted"] = point_std[i, j]
                        main.append(row)
            if block_length != block:
                continue
            parts = point[..., 1:]
            unknown_point = array_counts(supported, STRATA, airports, cells, ["N", "U", "D_unknown"])
            for j, metric in enumerate(METRICS):
                for ai, bi in ab:
                    comp = np.dot((cell_rate[ai, :, j]+cell_rate[bi, :, j])/2,
                                  by_airport[ai]-by_airport[bi])
                    within = np.dot((by_airport[ai]+by_airport[bi])/2,
                                    cell_rate[ai, :, j]-cell_rate[bi, :, j])
                    gap = point_crude[ai, j]-point_crude[bi, j]
                    if abs(comp + within - gap) > 1e-12:
                        raise ValueError("Kitagawa identity failed")
                    decompositions.append({
                        "minimum": floor, "metric": metric, "a": airports[ai], "b": airports[bi],
                        "common_crude_gap_pp": 100*gap, "composition_pp": 100*comp,
                        "within_pp": 100*within,
                        "audited_events_a": int(parts[ai, :, j].sum()),
                        "audited_events_b": int(parts[bi, :, j].sum()),
                    })
            weights = reference(point)
            upper = point.copy()
            upper[..., 1:] += unknown_point[..., 1:]
            for pos, metric in enumerate(METRICS):
                low = rates(point, weights)[:, pos]
                high = rates(upper, weights)[:, pos]
                direction = 1 if metric == "R1" else -1
                for i, airport in enumerate(airports):
                    if high[i] == low[i]:
                        continue
                    other = np.arange(len(airports)) != i
                    bounds.append({"airport": airport, "minimum": floor, "metric": metric,
                                   "adjusted_lower": low[i], "adjusted_upper": high[i],
                                   "width_pp": 100*(high[i]-low[i]),
                                   "rank_at_lower": 1 + int(((low[other]-low[i])*direction > 0).sum()),
                                   "rank_at_upper": 1 + int(((low[other]-high[i])*direction > 0).sum())})
    return tuple(pd.DataFrame(items) for items in (main, pairs, rankings, decompositions, bounds))


def direct_special(data, strata, airports, base, minimum=1):
    support, cells = shared(data, strata, airports, minimum)
    counts = array_counts(support, strata, airports, cells, ["N", "T", "D"])
    adjusted = rates(counts, reference(counts))
    crude = rates(counts)
    table = pd.DataFrame({"airport": airports, "cells": len(cells), "common_n": counts[..., 0].sum(axis=1),
                          "R1_common": crude[:, 0], "D60_common": crude[:, 1],
                          "R1_adjusted": adjusted[:, 0], "D60_adjusted": adjusted[:, 1]})
    table["coverage"] = table.common_n / table.airport.map(base)
    return table, support, cells


def seasonal_pattern(daily, dates, draws):
    data = daily.loc[daily.airport.isin(PAIR) & daily.season.isin(["DJF", "JJA"])].copy()
    data["year"] = data.date.dt.year
    full = data.groupby(["airport", "year", "season"], observed=True).N.sum()
    profiles = []
    boot_all = []
    strata_for = {"full": ["all"], "time_distance": ["time", "band"],
                  "common_carrier": ["carrier"], "matched_route_carrier_time": ["dest", "carrier", "time"]}
    data["all"] = 1
    seasons = ("DJF", "JJA")
    years = list(sorted(data.year.unique()))
    columns = ["N", "T", "D", "C", "V", "L60"]
    for scenario, strata in strata_for.items():
        table = data.groupby([*strata, "airport", "year", "season"], observed=True).N.sum().unstack(
            ["airport", "year", "season"], fill_value=0)
        floor = 50 if scenario == "matched_route_carrier_time" else 1
        cells = table.index[(table >= floor).all(axis=1)].to_frame(index=False)
        support = data.merge(cells, on=strata)
        pooled = array_counts(support, strata, PAIR, cells, columns)
        weights = reference(pooled)
        for year in ["pooled", *years]:
            groups = support if year == "pooled" else support.loc[support.year.eq(year)]
            for season in seasons:
                group = groups.loc[groups.season.eq(season)]
                point = array_counts(group, strata, PAIR, cells, columns)
                rep = bootstrap_counts(group, strata, PAIR, cells, dates, draws, columns)
                crude = rates(point)
                point_adjusted = rates(point, weights)
                margin = (point_adjusted[1] - point_adjusted[0]) * 100
                boot = rates(rep, weights)
                boot_all.append((boot[:, 1, :] - boot[:, 0, :]) * 100)
                for idx, airport in enumerate(PAIR):
                    total = full.loc[airport].xs(season, level="season")
                    total = total.sum() if year == "pooled" else total.loc[year]
                    profiles.append({"scenario": scenario, "year": year, "season": season,
                                     "airport": airport, "cells": len(cells), "n": int(point[idx, :, 0].sum()),
                                     "coverage": point[idx, :, 0].sum()/total,
                                     **{f"{metric}_same_sample_crude": crude[idx, j]
                                        for j, metric in enumerate(("R1","D60","RC","RV","R_L60"))},
                                     **{metric: point_adjusted[idx, j]
                                        for j, metric in enumerate(("R1","D60","RC","RV","R_L60"))},
                                     **{f"{metric}_sfo_minus_atl_pp": margin[j]
                                        for j, metric in enumerate(("R1","D60","RC","RV","R_L60"))}})
    profile = pd.DataFrame(profiles)
    grid = profile.loc[profile.airport.eq("SFO")].reset_index(drop=True)
    point_boot = np.stack(boot_all, axis=1)
    winter = np.flatnonzero(grid.season.eq("DJF"))
    summer = np.flatnonzero(grid.season.eq("JJA"))
    results = []
    for j, metric in enumerate(METRICS):
        points = grid[f"{metric}_sfo_minus_atl_pp"].to_numpy()
        values = point_boot[:, :, j]
        swings = points[summer] - points[winter]
        lo, hi, valid = intervals(np.r_[points, swings], np.c_[values, values[:, summer]-values[:, winter]])
        for k, (w, s) in enumerate(zip(winter, summer)):
            results.append({"scenario": grid.loc[w, "scenario"], "year": grid.loc[w, "year"],
                            "metric": metric, "winter_gap_pp": points[w], "winter_lo_pp": lo[w],
                            "winter_hi_pp": hi[w], "summer_gap_pp": points[s], "summer_lo_pp": lo[s],
                            "summer_hi_pp": hi[s], "swing_pp": swings[k], "swing_lo_pp": lo[len(points)+k],
                            "swing_hi_pp": hi[len(points)+k], "bootstrap_valid": valid,
                            "supported_reversal": (lo[w] > 0 and hi[s] < 0) or (hi[w] < 0 and lo[s] > 0)})
    audit = add_rates(data.groupby(["airport", "year", "season", "carrier"], observed=True)[COUNTS].sum().reset_index())
    audit["flight_share"] = audit.N / audit.groupby(["airport", "year", "season"]).N.transform("sum")
    return profile, pd.DataFrame(results), audit


def summaries(daily, full, q1):
    reference_counts = dict(N="n", C="n_C", V="n_V", T="n_T", U="n_U",
                            D="n_severe_60", D_unknown="n_severe_60_unknown")
    monthly = daily.assign(month=daily.date.dt.strftime("%Y-%m"))
    monthly = monthly.groupby(["month", "airport"], observed=True)[COUNTS].sum().sort_index()
    q1_month = pd.read_csv(q1 / "monthly_metrics.csv").set_index(["month", "airport"]).reindex(monthly.index)
    for q2, q1_name in reference_counts.items():
        if not np.array_equal(monthly[q2], q1_month[q1_name]):
            raise ValueError(f"Q1 monthly reference differs: {q2}")
    q1_full = pd.read_csv(q1 / "airport_metrics.csv").set_index("airport").reindex(full.index)
    if not np.array_equal(full[list(reference_counts)], q1_full[list(reference_counts.values())]):
        raise ValueError("Q1 full reference differs")
    dates = add_rates(daily.groupby(["date", "airport"], observed=True)[COUNTS].sum().reset_index())
    monthly = add_rates(monthly.reset_index())
    results = []
    for label, basis in (("daily", dates), ("monthly", monthly)):
        for airport, table in basis.groupby("airport"):
            for metric in METRICS:
                q10, q25, med, q75, q90 = table[metric].quantile([.1,.25,.5,.75,.9])
                results.append({"period": label, "airport": airport, "metric": metric, "periods": len(table),
                                "p10": q10, "p25": q25, "median": med, "p75": q75, "p90": q90, "IQR": q75-q25})
    return monthly, dates, pd.DataFrame(results)


def run(args):
    from q2_figures import render

    located = selected_months(args.analysis_dir, args.years)
    daily, scope, sources = read_counts(located)
    airport = daily.groupby("airport", observed=True)[COUNTS].sum().reindex(AIRPORTS)
    if airport.N.sum() != scope[0] + scope[1]:
        raise ValueError("Outbound/overlap reconciliation failed")
    if sorted(args.years) == [2023, 2024] and scope.tolist() != [435038, 2812313, 2812205]:
        raise ValueError("Development scope differs from audited cross-tab")
    dates = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{year}-01-01", f"{year}-12-31").to_numpy()
        for year in sorted(args.years)]))
    draws = bootstrap_weights(dates, args.block_days, args.bootstrap_reps, args.seed)
    month, day, variation = summaries(daily, airport, args.q1_dir)
    pooled_keys = [k for k in KEYS if k != "date"]
    pooled = daily.groupby(pooled_keys, observed=True)[COUNTS].sum().reset_index()
    print("Verified Q1 counts in every airport-month; estimating standardized rankings", flush=True)
    main, pairs, ranks, kita, bounds = main_comparison(
        daily, dates, draws, args.block_days, args.bootstrap_reps, args.seed)

    described = []
    for dimension in (*STRATA, "carrier", "dest"):
        grouped = add_rates(pooled.groupby(["airport", dimension], observed=True)[COUNTS].sum().reset_index())
        grouped = grouped.rename(columns={dimension: "level"})
        grouped.insert(1, "dimension", dimension)
        grouped["flight_share"] = grouped.N / grouped.airport.map(airport.N)
        for metric in METRICS:
            grouped[f"{metric}_rank"] = grouped.groupby("level")[metric].rank(
                ascending=metric == "D60", method="min")
        described.append(grouped)
    descriptive = pd.concat(described, ignore_index=True)
    carrier_matrix = pooled.groupby(["carrier", "airport"], observed=True).N.sum().unstack(fill_value=0)
    carriers = carrier_matrix.index[(carrier_matrix >= 1000).all(axis=1)]
    carrier_strata = [*STRATA, "dow", "carrier"]
    car = pooled.loc[pooled.carrier.isin(carriers)]
    carrier, car_support, car_cells = direct_special(car, carrier_strata, AIRPORTS, airport.N)
    carrier.insert(0, "specialty", "carrier_joint")
    destination, dest_support, dest_cells = direct_special(pooled, ["dest"], AIRPORTS, airport.N)
    destination.insert(0, "specialty", "destination_all")
    pair_dest = ("ATL", "ORD")
    pair_keys = ["season", "time", "dest"]
    by_pair, _, _ = direct_special(pooled, pair_keys, pair_dest, airport.N, 50)
    by_pair.insert(0, "specialty", "destination_ATL_ORD")
    destination_time_sample, destination_time = shared(pooled, ["dest", "season", "time"], AIRPORTS)
    destination_band_sample, destination_band = shared(pooled, ["dest", *STRATA], AIRPORTS)
    combined_sample, combined = shared(car.loc[car.dest.isin(dest_cells.dest)], [*carrier_strata, "dest"], AIRPORTS)
    support_audit = pd.concat([
        support_summary(sample, cells, airport.N, label)
        for sample, cells, label in (
            (car, carriers, "common_carrier_marginal"), (car_support, car_cells, "carrier_joint"),
            (dest_support, dest_cells, "destination_all"),
            (destination_time_sample, destination_time, "destination_season_time"),
            (destination_band_sample, destination_band, "destination_season_time_band"),
            (combined_sample, combined, "carrier_destination_joint"),
        )
    ], ignore_index=True)
    if sorted(args.years) == [2023, 2024] and (len(combined), len(destination_band)) != (0, 0):
        raise ValueError("Development joint-support check differs from audit")
    print("Specialties estimated; resampling SFO–ATL winter and summer", flush=True)
    pattern, pattern_ci, reporting = seasonal_pattern(daily, dates, draws)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "airport_adjustment": main, "pairwise_inference": pairs, "rank_sets": ranks,
        "kitagawa": kita, "four_dimensions": descriptive, "daily_rates": day, "monthly_rates": month,
        "variation": variation, "specialties": pd.concat([carrier, destination, by_pair], ignore_index=True),
        "support_audit": support_audit, "unknown_bounds": bounds,
        "sfo_atl_profiles": pattern,
        "sfo_atl_inference": pattern_ci, "sfo_atl_reporting": reporting,
    }
    for name, table in results.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g", encoding="utf-8-sig")
    primary = pairs.loc[pairs.minimum.eq(1) & pairs.weights.eq("pooled") & pairs.block.eq(args.block_days)]
    figure_paths = render(results, primary, args.figures)
    checks = {
        "q1_equal_airports": len(airport), "q1_equal_airport_months": len(month),
        "kitagawa_pair_identities": len(kita), "days": len(dates),
        "outbound": int(scope[0]+scope[1]), "inbound": int(scope[0]+scope[2]),
        "shared": int(scope[0]), "outbound_only": int(scope[1]), "inbound_only": int(scope[2]),
        "carriers": list(carriers), "carrier_joint_cells": len(car_cells),
        "common_destination_cells": len(dest_cells), "destination_season_time_cells": len(destination_time),
        "destination_season_time_band_cells": len(destination_band), "carrier_destination_joint_cells": len(combined),
        "min_rank_family_bootstrap_valid": int(pairs.bootstrap_valid.min()),
        "min_pattern_family_bootstrap_valid": int(pattern_ci.bootstrap_valid.min()),
        "unknown_by_airport": airport.loc[airport.U.gt(0), "U"].astype(int).to_dict(),
    }
    manifest = {"years": args.years, "bootstrap_reps": args.bootstrap_reps, "block_days": args.block_days,
                "seed": args.seed, "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                "checks": checks, "inputs": sources,
                "script_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("q2_figures.py"))},
                "outputs": {name+".csv": sha256(output / (name+".csv")) for name in results},
                "figures": {p.name: sha256(p) for p in figure_paths}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--analysis-dir", type=Path, nargs="+", default=[ROOT/"data/analysis/development"])
    parser.add_argument("--q1-dir", type=Path, default=ROOT/"outputs/问题一计算结果")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/问题二计算结果")
    parser.add_argument("--figures", type=Path, default=ROOT/"figures/问题二计算结果")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if len(args.years) != len(set(args.years)) or args.bootstrap_reps < 2:
        parser.error("Years must be distinct and at least two repetitions are required")
    return args


if __name__ == "__main__":
    run(parse_args())
