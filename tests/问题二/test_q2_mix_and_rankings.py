import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "问题二"))
from q2_mix_and_rankings import (array_counts, bootstrap_counts, bootstrap_weights,
                                 intervals, rates, reference, shared)


def test_common_support_sorting_and_bootstrap_preserve_matched_counts():
    dates = pd.date_range("2024-02-28", periods=3)
    flights = pd.DataFrame({
        "airport": ["B", "A", "B", "A", "A"],
        "time": [1, 1, 0, 0, 2],
        "date": dates[[1, 0, 1, 0, 2]],
        "N": [10, 15, 30, 25, 100],
        "T": [10, 0, 0, 25, 100],
    })
    sample, cells = shared(flights, ["time"], ["A", "B"])
    point = array_counts(sample, ["time"], ["A", "B"], cells, ["N", "T"])
    assert cells.time.tolist() == [0, 1]
    np.testing.assert_allclose(rates(point).ravel(), [25/40, 10/40])
    np.testing.assert_allclose(rates(point, reference(point)).ravel(), [55/80, 25/80])
    reps = bootstrap_counts(sample, ["time"], ["A", "B"], cells, dates,
                            np.array([[1, 1, 1], [2, 1, 0]], dtype=float), ["N", "T"])
    np.testing.assert_array_equal(reps[0], point)
    np.testing.assert_array_equal(reps[1, 0], 2*point[0])
    np.testing.assert_array_equal(reps[1, 1], point[1])


def test_bootstrap_preserves_calendar_segments_and_is_reproducible():
    dates = pd.date_range("2023-12-27", "2024-03-07")
    first = bootstrap_weights(dates, 7, 30, 2026)
    np.testing.assert_array_equal(first, bootstrap_weights(dates, 7, 30, 2026))
    labels = [(dates.year == 2023), ((dates.year == 2024) & (dates.month <= 2)), (dates.month == 3)]
    for mask in labels:
        np.testing.assert_array_equal(first[:, mask].sum(axis=1), mask.sum())


def test_simultaneous_family_reports_number_of_complete_draws():
    points = np.array([1., 2.])
    draws = np.array([[1., 2.], [1.3, 1.8], [.9, 2.2], [np.nan, 2.]])
    lower, upper, n = intervals(points, draws)
    assert n == 3
    np.testing.assert_allclose(lower + upper, 2*points)
