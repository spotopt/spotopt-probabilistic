"""Tests for features.add_lags."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spotopt._features import add_lags

COL_NAMES = ["obs"]


@pytest.mark.parametrize(
    ("df_in", "df_expected", "lag_days"),
    [
        (
            pd.DataFrame(
                {"obs": [0.0, 1.0]},
                index=pd.DatetimeIndex(
                    [
                        dt.datetime(2025, 1, 1, 1, tzinfo=None),
                        dt.datetime(2025, 1, 2, 1, tzinfo=None),
                    ],
                    name="delivery",
                ),
            ),
            pd.DataFrame(
                {
                    "obs": [0.0, 1.0],
                    "obs_lag_1d": [np.nan, 0.0],
                },
                index=pd.DatetimeIndex(
                    [
                        dt.datetime(2025, 1, 1, 1, tzinfo=None),
                        dt.datetime(2025, 1, 2, 1, tzinfo=None),
                    ],
                    name="delivery",
                ),
            ),
            1,
        ),
        (
            pd.DataFrame(
                {"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        dt.datetime(2025, 1, 1, 1, tzinfo=None),
                        dt.datetime(2025, 1, 2, 1, tzinfo=None),
                        dt.datetime(2025, 1, 3, 1, tzinfo=None),
                    ],
                    name="delivery",
                ),
            ),
            pd.DataFrame(
                {
                    "obs": [0.0, 1.0, 2.0],
                    "obs_lag_2d": [np.nan, np.nan, 0.0],
                },
                index=pd.DatetimeIndex(
                    [
                        dt.datetime(2025, 1, 1, 1, tzinfo=None),
                        dt.datetime(2025, 1, 2, 1, tzinfo=None),
                        dt.datetime(2025, 1, 3, 1, tzinfo=None),
                    ],
                    name="delivery",
                ),
            ),
            2,
        ),
    ],
)
def test_standard_use_cases(
    df_in: pd.DataFrame,
    df_expected: pd.DataFrame,
    lag_days: int,
) -> None:
    """Test standard use cases."""
    df_out = add_lags(df_in, COL_NAMES, lag_days=lag_days)
    assert_frame_equal(df_out, df_expected)


@pytest.mark.parametrize("lag_days", [0, -1, -2])
def test_lag_days_positive(lag_days: int) -> None:
    """Lag days must be positive."""
    df = pd.DataFrame(
        {"obs": [0.0]},
        index=pd.DatetimeIndex(
            [dt.datetime(2025, 1, 1, 1, tzinfo=None)],
            name="delivery",
        ),
    )
    msg = "lag_days needs to be a positive integer."
    with pytest.raises(ValueError, match=msg):
        add_lags(df, COL_NAMES, lag_days=lag_days)


@pytest.mark.parametrize("lag_days", [2.1, "2", 0.9, 1.0])
def test_lag_days_integer(lag_days: int) -> None:
    """Lag days must be integer."""
    df = pd.DataFrame(
        {"obs": [0.0]},
        index=pd.DatetimeIndex(
            [dt.datetime(2025, 1, 1, 1, tzinfo=None)],
            name="delivery",
        ),
    )
    msg = "lag_days needs to be an integer."
    with pytest.raises(TypeError, match=msg):
        add_lags(df, COL_NAMES, lag_days=lag_days)


@pytest.mark.parametrize(
    "case",
    [
        (True, ["obs_lag_1d"]),
        (False, ["obs", "obs_lag_1d"]),
    ],
)
def test_drop_origin(case: tuple[bool, list[str]]) -> None:
    """Test dropping the origin column."""
    drop_origin, expected_cols = case
    df_in = pd.DataFrame(
        {"obs": [0.0]},
        index=pd.DatetimeIndex(
            [dt.datetime(2025, 1, 1, 1, tzinfo=None)],
            name="delivery",
        ),
    )
    df_out = add_lags(df_in, ["obs"], lag_days=1, drop_origin=drop_origin)
    assert list(df_out.columns) == expected_cols
