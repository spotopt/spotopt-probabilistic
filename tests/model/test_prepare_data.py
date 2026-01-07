"""Tests for model._prepare_data."""

import pandas as pd

from spotopt._types import Frequency
from spotopt.model import _prepare_data


def test_standard_use_cases():
    """Test standard use cases for _prepare_data."""
    df_in = pd.DataFrame(
        {
            "obs": range(48),
            "fcast": range(48, 96),
        },
        index=pd.date_range(
            start=pd.Timestamp("2025-01-02 00:00:00", tz="CET"),
            end=pd.Timestamp("2025-01-03 23:00:00", tz="CET"),
            freq="60min",
            name="delivery",
        ),
    )
    df_out = _prepare_data(df_in, frequency=Frequency(60))
    assert isinstance(df_out, pd.DataFrame)
    assert df_out["obs_lag_1d"].to_list() == list(range(24))
    assert all(df_out["obs_min_lag_1d"] == 0.0)
    assert all(df_out["obs_max_lag_1d"] == 23.0)  # noqa: PLR2004
    assert df_out.columns.to_list() == [
        "obs",
        "fcast",
        "obs_lag_1d",
        "obs_min_lag_1d",
        "obs_max_lag_1d",
        "weekday_1",
        "weekday_2",
        "weekday_3",
        "weekday_4",
        "weekday_5",
        "weekday_6",
        "weekday_7",
        "hour",
        "minute",
    ]
    assert df_out.shape == (24, 14)
