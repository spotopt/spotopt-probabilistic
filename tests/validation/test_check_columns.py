"""Tests for function _check_columns."""

from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import pytest

from spotopt._exceptions import MissingColumnsError
from spotopt._validation import _check_columns


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
            ),
            does_not_raise(),
        ),
        (
            pd.DataFrame(
                data={"obs": [0.0], "fcast": [0.1], "extra": [1.0]},
            ),
            does_not_raise(),
        ),
        (
            pd.DataFrame(
                data={"obs": [0.0]},
            ),
            pytest.raises(
                MissingColumnsError,
                match="Missing required columns:",
            ),
        ),
        (
            pd.DataFrame(
                data={"fcast": [0.1]},
            ),
            pytest.raises(
                MissingColumnsError,
                match="Missing required columns:",
            ),
        ),
    ],
)
def test_check_columns(df_in: pd.DataFrame, expectation) -> None:
    """Test required column validation."""
    with expectation:
        _check_columns(df_in)
