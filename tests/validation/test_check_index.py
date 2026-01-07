"""Tests for function _check_index."""

import re
from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import pytest

from spotopt._exceptions import IndexNameError
from spotopt._validation import _check_index


@contextmanager
def does_not_raise() -> Generator:
    """Utility context manager signalling no exception."""
    yield


@pytest.mark.parametrize(
    ("df_in", "expectation"),
    [
        (
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [0.1]},
                index=pd.date_range(
                    start=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                    periods=1,
                    freq="60min",
                    name="delivery",
                ),
            ),
            does_not_raise(),
        ),
        (
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [0.1]},
                index=pd.date_range(
                    start=pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                    periods=1,
                    freq="60min",
                    name="wrong_name",
                ),
            ),
            pytest.raises(
                IndexNameError,
                match=re.escape("Index must be named 'delivery'."),
            ),
        ),
        (
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [0.1]},
                index=pd.Index([0], name="delivery"),
            ),
            pytest.raises(
                TypeError,
                match=re.escape("Index must be a pandas DatetimeIndex."),
            ),
        ),
        (
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [0.1]},
                index=pd.date_range(
                    start=pd.Timestamp("2025-01-01 00:00:00"),
                    periods=1,
                    freq="60min",
                    name="delivery",
                ),
            ),
            pytest.raises(
                ValueError,
                match=re.escape("Index must have time zone 'CET'."),
            ),
        ),
    ],
)
def test_check_index(df_in: pd.DataFrame, expectation) -> None:
    """Test index validation."""
    with expectation:
        _check_index(df_in)
