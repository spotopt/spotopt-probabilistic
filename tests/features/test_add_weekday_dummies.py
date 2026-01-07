"""Tests for function add_weekday_dummies."""

import pandas as pd
import pytest

from spotopt._features import add_weekday_dummies

REQUIRED_COLS = [f"weekday_{i}" for i in range(1, 8)]


@pytest.mark.parametrize(
    "df_in",
    [
        pd.DataFrame(
            {"obs": [0.0]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-01-01 01:00")],
                name="delivery",
            ),
        ),
        pd.DataFrame(
            {"obs": [0.0, 1.0, 2.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-01-01 01:00"),
                    pd.Timestamp("2025-01-02 01:00"),
                    pd.Timestamp("2025-01-03 01:00"),
                ],
                name="delivery",
            ),
        ),
    ],
)
def test_all_weekdays_present(df_in: pd.DataFrame) -> None:
    """Test that all weekday dummy columns are present."""
    df_out = add_weekday_dummies(df_in)
    assert set(REQUIRED_COLS).issubset(df_out.columns)
    assert df_out.index.equals(df_in.index)


@pytest.mark.parametrize(
    "df_in",
    [
        pd.DataFrame(
            {"obs": [0.0]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2025-01-01 01:00")],
                name="delivery",
            ),
        ),
        pd.DataFrame(
            {"obs": [0.0, 1.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2024-12-30 01:00"),
                    pd.Timestamp("2025-01-01 01:00"),
                ],
                name="delivery",
            ),
        ),
    ],
)
def test_all_weekday_order(df_in: pd.DataFrame) -> None:
    """Test that weekday dummy columns follow expected order."""
    df_out = add_weekday_dummies(df_in)
    assert list(df_out.columns) == [
        "obs",
        *sorted(REQUIRED_COLS),
    ]
