"""Tests for function _check_delivery_and_end."""

import re
from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import pytest

from spotopt._types import Frequency
from spotopt._validation import _check_delivery_begin_and_end


@contextmanager
def does_not_raise() -> Generator:
    """Util function to check if no error is raised."""
    yield


@pytest.mark.parametrize(
    ("df_in", "frequency", "expectation"),
    [
        # Hourly: begin and end are correct.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 23:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            does_not_raise(),
        ),
        # Quarter-hourly: begin and end are correct.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 23:45:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(15),
            does_not_raise(),
        ),
        # Hourly: wrong begin.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:30:00", tz="CET"),
                        pd.Timestamp("2025-01-01 23:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "First delivery time step must be at the start of a day.",
                ),
            ),
        ),
        # Hourly: wrong end.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                        pd.Timestamp("2025-01-02 22:00:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(60),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Last delivery time step must be at the end of a day.",
                ),
            ),
        ),
        # Quarter-hourly: wrong begin.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:30:00", tz="CET"),
                        pd.Timestamp("2025-01-01 23:45:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(15),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "First delivery time step must be at the start of a day.",
                ),
            ),
        ),
        # Quarter-hourly: wrong end.
        (
            pd.DataFrame(
                data={"obs": [0, 1]},
                index=pd.DatetimeIndex(
                    name="delivery",
                    data=[
                        pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 23:30:00", tz="CET"),
                    ],
                ),
            ),
            Frequency(15),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Last delivery time step must be at the end of a day.",
                ),
            ),
        ),
    ],
)
def test_standard_use_cases(df_in, frequency, expectation):
    """Test standard use cases."""
    with expectation:
        _check_delivery_begin_and_end(df_in, frequency=frequency)
