"""Tests for function convert_from_none_time_zone."""

from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spotopt._utils import convert_from_none_time_zone


@patch("spotopt._constants.TZ_STR", "CET")
@pytest.mark.parametrize(
    ("index_in", "index_expected"),
    [
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-01-01 00:00",
                    "2025-01-01 23:00",
                    freq="h",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-01-01 00:00",
                    "2025-01-01 23:00",
                    freq="h",
                    tz="CET",
                ),
            ),
            id="none-to-cet-hourly",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-01-01 00:00",
                    "2025-01-01 23:45",
                    freq="15min",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-01-01 00:00",
                    "2025-01-01 23:45",
                    freq="15min",
                    tz="CET",
                ),
            ),
            id="none-to-cet-quarter-hourly",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-03-30 00:00",
                    "2025-03-30 23:00",
                    freq="h",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-03-30 00:00",
                    "2025-03-30 23:00",
                    freq="h",
                    tz="CET",
                ),
            ),
            id="none-to-cet-hourly-dst-march",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-03-30 00:00",
                    "2025-03-30 23:45",
                    freq="15min",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-03-30 00:00",
                    "2025-03-30 23:45",
                    freq="15min",
                    tz="CET",
                ),
            ),
            id="none-to-cet-quarter-hourly-dst-march",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-10-26 00:00",
                    "2025-10-26 23:00",
                    freq="h",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-10-26 00:00",
                    "2025-10-26 23:00",
                    freq="h",
                    tz="CET",
                ),
            ),
            id="none-to-cet-hourly-dst-october",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range(
                    "2025-10-26 00:00",
                    "2025-10-26 23:45",
                    freq="15min",
                    tz=None,
                ),
            ),
            pd.DataFrame(
                index=pd.date_range(
                    "2025-10-26 00:00",
                    "2025-10-26 23:45",
                    freq="15min",
                    tz="CET",
                ),
            ),
            id="none-to-cet-quarter-hourly-dst-october",
        ),
    ],
)
def test_convert_from_none_time_zone(index_in, index_expected):
    """Test standard behavior."""
    index_out = convert_from_none_time_zone(index_in)
    assert_frame_equal(index_out, index_expected)
