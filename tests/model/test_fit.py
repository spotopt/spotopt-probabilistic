"""Tests for model._fit."""

from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.linear_model import QuantileRegressor

from spotopt import ModelName, SpotOptConfig
from spotopt._types import Frequency
from spotopt.model import _fit

_QUANTILES = [5, 25, 50, 75, 95]


@patch("spotopt._constants.QUANTILES", _QUANTILES)
@patch("spotopt._constants.MIN_NR_DAYS_TRAIN", 0)
@pytest.mark.parametrize(
    "frequency",
    [
        Frequency(60),
        Frequency(15),
    ],
)
def test_standard_use_cases(frequency: Frequency) -> None:
    """Test standard use cases for _fit."""
    freq_multiplier = 60 // frequency.value
    start = pd.Timestamp("2025-01-02 00:00:00", tz="CET")
    end = (
        start + pd.DateOffset(days=2) - pd.DateOffset(minutes=frequency.value)
    )
    df_in = pd.DataFrame(
        {
            "obs": range(48 * freq_multiplier),
            "fcast": range(48 * freq_multiplier, 96 * freq_multiplier),
        },
        index=pd.date_range(
            start=start,
            end=end,
            freq=f"{frequency.value}min",
            name="delivery",
        ),
    )
    config = SpotOptConfig(
        model_name=ModelName("Lasso"),
        frequency=frequency,
        mdl_kwargs={"alpha": 0.1},
    )
    fit_cols, qrs = _fit(df_in, config=config)
    assert isinstance(fit_cols, list)
    assert isinstance(qrs, dict)
    assert len(qrs) == 24 * freq_multiplier * len(_QUANTILES)
    for mdl in qrs.values():
        assert isinstance(mdl, QuantileRegressor)
