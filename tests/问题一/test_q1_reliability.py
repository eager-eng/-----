from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "问题一"))
from q1_reliability import timing_bounds


def test_bounds_keep_verified_arrivals_with_unknown_delay():
    frame = pd.DataFrame({
        "outcome_code": ["T", "T", "V", "C", "U"],
        "final_reached": ["True", "True", "True", "False", ""],
        "ever_diverted": ["False", "False", "True", "False", ""],
        "undiverted_on_time": ["True", "True", "False", "False", ""],
        "severe_45": ["False", "False", "True", "True", ""],
        "severe_60": ["False", "False", "True", "True", ""],
        "severe_90": ["False", "False", "True", "True", ""],
        "final_late_60": ["False", "False", "", "False", ""],
    })
    missing = np.full(len(frame), np.nan)
    strict_low, strict_high, wide_low, wide_high = timing_bounds(
        frame, frame.outcome_code.eq("T").to_numpy(), missing,
        np.array([0, np.nan, np.nan, np.nan, np.nan]), missing)

    np.testing.assert_array_equal(strict_low, [-np.inf, -np.inf, np.inf, np.inf, -np.inf])
    np.testing.assert_array_equal(strict_high, [0, 14, np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(wide_low, [-np.inf, -np.inf, -np.inf, np.inf, -np.inf])
    np.testing.assert_array_equal(wide_high, [0, 14, np.inf, np.inf, np.inf])
