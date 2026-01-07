"""Tests for function account_for_dst."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spotopt._types import Frequency
from spotopt._utils import account_for_dst


@pytest.mark.parametrize(
    ("df_in", "df_expected", "frequency"),
    [
        # No DST, hourly.
        (
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-01-01 00:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 01:00:00", tz="CET"),
                        pd.Timestamp("2025-01-01 02:00:00", tz="CET"),
                    ],
                ),
            ),
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-01-01 00:00:00", tz=None),
                        pd.Timestamp("2025-01-01 01:00:00", tz=None),
                        pd.Timestamp("2025-01-01 02:00:00", tz=None),
                    ],
                ),
            ),
            Frequency(60),
        ),
        # No DST, quarter-hourly.
        (
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-01-01 00:45:00", tz="UTC"),
                        pd.Timestamp("2025-01-01 01:00:00", tz="UTC"),
                        pd.Timestamp("2025-01-01 01:15:00", tz="UTC"),
                    ],
                ),
            ),
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-01-01 01:45:00"),
                        pd.Timestamp("2025-01-01 02:00:00"),
                        pd.Timestamp("2025-01-01 02:15:00"),
                    ],
                ),
            ),
            Frequency(15),
        ),
        # Last Sunday in March (23 hours), hourly.
        (
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-03-30 00:00:00", tz="UTC"),
                        pd.Timestamp("2025-03-30 01:00:00", tz="UTC"),
                        pd.Timestamp("2025-03-30 02:00:00", tz="UTC"),
                    ],
                ),
            ),
            pd.DataFrame(
                data={"obs": [0.0, 0.5, 1.0, 2.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-03-30 01:00:00"),
                        pd.Timestamp("2025-03-30 02:00:00"),
                        pd.Timestamp("2025-03-30 03:00:00"),
                        pd.Timestamp("2025-03-30 04:00:00"),
                    ],
                ),
            ),
            Frequency(60),
        ),
        # Last Sunday in March (23 hours), quarter-hourly.
        (
            pd.DataFrame(
                data={"obs": [float(value) for value in range(9)]},
                index=pd.date_range(
                    start="2025-03-30 00:00:00",
                    end="2025-03-30 02:00:00",
                    freq="15min",
                    tz="UTC",
                ),
            ),
            pd.DataFrame(
                data={
                    "obs": [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        3.2,  # Interpolated hour 2
                        3.4,  # Interpolated hour 2
                        3.6,  # Interpolated hour 2
                        3.8,  # Interpolated hour 2
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                    ],
                },
                index=pd.date_range(
                    start="2025-03-30 01:00:00",
                    end="2025-03-30 04:00:00",
                    freq="15min",
                ),
            ),
            Frequency(15),
        ),
        # Last Sunday in October (24 hours), hourly.
        (
            pd.DataFrame(
                data={"obs": [0.0, 1.0, 2.0, 3.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-10-26 00:00:00", tz="UTC"),
                        pd.Timestamp("2025-10-26 01:00:00", tz="UTC"),
                        pd.Timestamp("2025-10-26 02:00:00", tz="UTC"),
                        pd.Timestamp("2025-10-26 03:00:00", tz="UTC"),
                    ],
                ),
            ),
            pd.DataFrame(
                data={"obs": [0.5, 2.0, 3.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2025-10-26 02:00:00"),
                        pd.Timestamp("2025-10-26 03:00:00"),
                        pd.Timestamp("2025-10-26 04:00:00"),
                    ],
                ),
            ),
            Frequency(60),
        ),
        # Last Sunday in October (24 hours), quarter-hourly.
        (
            pd.DataFrame(
                data={"obs": [float(value) for value in range(12)]},
                index=pd.date_range(
                    start="2025-10-26 00:00:00",
                    end="2025-10-26 02:45:00",
                    freq="15min",
                    tz="UTC",
                ),
            ),
            pd.DataFrame(
                data={
                    "obs": [
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                    ],
                },
                index=pd.date_range(
                    start="2025-10-26 02:00:00",
                    end="2025-10-26 03:45:00",
                    freq="15min",
                ),
            ),
            Frequency(15),
        ),
    ],
)
def test_standard_use_cases(df_in, df_expected, frequency):
    """Test standard use cases."""
    df_out = account_for_dst(df_in, frequency)
    assert_frame_equal(df_out, df_expected, check_freq=False)
