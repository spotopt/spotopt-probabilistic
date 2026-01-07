"""Tests for model.SpotOptModel."""

from unittest.mock import patch

import pandas as pd

from spotopt import ModelName, SpotOptConfig, SpotOptModel
from spotopt._types import Frequency

_QUANTILES = [5, 25, 50, 75, 95]


@patch("spotopt._constants.QUANTILES", _QUANTILES)
@patch("spotopt._constants.MIN_NR_DAYS_TRAIN", 0)
def test_standard_use_cases() -> None:
    """Test standard use cases for _fit."""
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
    config = SpotOptConfig(
        model_name=ModelName("Lasso"),
        frequency=Frequency(60),
        mdl_kwargs={"alpha": 0.1},
    )
    spotopt_mdl = SpotOptModel(config)
    spotopt_mdl.fit(df_in)
    assert hasattr(spotopt_mdl, "fit_cols")
    assert hasattr(spotopt_mdl, "qrs")
    predictions = spotopt_mdl.predict(df_in)
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (24, len(_QUANTILES))
