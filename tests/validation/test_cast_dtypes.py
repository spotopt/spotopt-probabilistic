"""Tests for function _cast_dtypes."""

import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spotopt._validation import _cast_dtypes


@pytest.mark.parametrize(
    ("df_in", "df_expected"),
    [
        (
            pd.DataFrame(
                data={"obs": [0], "fcast": [1]},
            ),
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [1.0]},
            ),
        ),
        (
            pd.DataFrame(
                data={"obs": [0], "fcast": [1], "extra": [2]},
            ),
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [1.0], "extra": [2.0]},
            ),
        ),
    ],
)
def test_cast_dtypes(
    df_in: pd.DataFrame,
    df_expected: pd.DataFrame,
) -> None:
    """Test dtype casting for base and additional columns."""
    df_out = _cast_dtypes(df_in)
    assert_frame_equal(df_out, df_expected)


def test_cast_dtypes_logs_additional_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that additional columns trigger an info log."""
    df_in = pd.DataFrame(
        data={"obs": [0], "fcast": [1], "extra": [2]},
    )
    with caplog.at_level(logging.INFO, logger="spotopt"):
        _cast_dtypes(df_in)
    assert "Casting additional columns to float" in caplog.text
