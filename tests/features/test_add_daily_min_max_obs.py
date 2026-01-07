"""Tests for function add_daily_min_max_obs."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spotopt._features import add_daily_min_max_obs


@pytest.mark.parametrize(
    ("df_in", "df_expected"),
    [
        (
            pd.DataFrame(
                {
                    "delivery": [
                        pd.Timestamp("2025-01-01 01:00:00"),
                    ],
                    "obs": [0.0],
                },
            ).set_index("delivery"),
            pd.DataFrame(
                {
                    "delivery": [
                        pd.Timestamp("2025-01-01 01:00:00"),
                    ],
                    "obs": [0.0],
                    "obs_min": [0.0],
                    "obs_max": [0.0],
                },
            ).set_index("delivery"),
        ),
        (
            pd.DataFrame(
                {
                    "delivery": [
                        pd.Timestamp("2025-01-01 01:00:00"),
                        pd.Timestamp("2025-01-02 01:00:00"),
                        pd.Timestamp("2025-01-02 02:00:00"),
                    ],
                    "obs": [0.0, 1.0, 2.0],
                },
            ).set_index("delivery"),
            pd.DataFrame(
                {
                    "delivery": [
                        pd.Timestamp("2025-01-01 01:00:00"),
                        pd.Timestamp("2025-01-02 01:00:00"),
                        pd.Timestamp("2025-01-02 02:00:00"),
                    ],
                    "obs": [0.0, 1.0, 2.0],
                    "obs_min": [0.0, 1.0, 1.0],
                    "obs_max": [0.0, 2.0, 2.0],
                },
            ).set_index("delivery"),
        ),
    ],
)
def test_standard_use_cases(df_in: pd.DataFrame, df_expected: pd.DataFrame):
    """Test daily min and max aggregation for the 'obs' column."""
    df_out = add_daily_min_max_obs(df_in)
    assert_frame_equal(df_out, df_expected, check_like=False)
