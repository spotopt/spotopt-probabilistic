"""Tests for function _check_gaps_in_delivery."""

import re
from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import pytest

from spotopt._types import Frequency
from spotopt._validation import _check_gaps_in_delivery


@contextmanager
def does_not_raise() -> Generator:
    """Util function to check if no error is raised."""
    yield


@pytest.mark.parametrize(
    ("df_in", "interval", "expectation"),
    [
        # No gaps, no errors.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 01:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 02:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            does_not_raise(),
        ),
        # Missing hour.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 01:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 03:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            pytest.raises(ValueError, match=re.escape("Missing time steps.")),
        ),
        # Unsorted.
        (
            pd.DataFrame(
                data={"obs": [1, 0]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 02:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 01:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            pytest.raises(ValueError, match=re.escape("Missing time steps.")),
        ),
    ],
)
def test_standard_use_cases(df_in, interval, expectation):
    """Test standard use cases."""
    with expectation:
        _check_gaps_in_delivery(df_in, frequency=interval)
