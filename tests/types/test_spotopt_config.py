"""Tests for _types.SpotOptConfig."""

import json
import re
from contextlib import nullcontext
from pathlib import Path

import pytest

from spotopt._exceptions import (
    ForbiddenKeyWordError,
    ParameterCombinationError,
)
from spotopt._types import Frequency, ModelName, SpotOptConfig

does_not_raise = nullcontext


@pytest.mark.parametrize(
    ("mdl_kwargs", "run_hyperparam_search", "expectation"),
    [
        (
            None,
            False,
            does_not_raise(),
        ),
        (
            {"alpha": 0.1},
            False,
            does_not_raise(),
        ),
        (
            {"alpha": 0.1},
            True,
            pytest.raises(
                ParameterCombinationError,
                match=("Choose either model kwargs or hyperparameter search"),
            ),
        ),
        (
            {"quantile": 0.5},
            False,
            pytest.raises(
                ForbiddenKeyWordError,
                match=re.escape(
                    "Model Lasso does not accept the following "
                    "keywords: {'quantile'}",
                ),
            ),
        ),
    ],
)
def test_spotopt_config_post_init(
    mdl_kwargs,
    run_hyperparam_search,
    expectation,
) -> None:
    """Test post_init checks."""
    with expectation:
        SpotOptConfig(
            model_name=ModelName("Lasso"),
            frequency=Frequency(60),
            mdl_kwargs=mdl_kwargs,
            run_hyperparam_search=run_hyperparam_search,
        )


def test_from_json(tmp_path: Path) -> None:
    """Test that from_json returns an equivalent SpotOptConfig."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "Lasso",
                "frequency": 60,
                "mdl_kwargs": {"alpha": 0.1},
            },
        ),
        encoding="utf-8",
    )

    cfg = SpotOptConfig.from_json(config_path)

    assert cfg == SpotOptConfig(
        model_name=ModelName("Lasso"),
        frequency=Frequency(60),
        mdl_kwargs={"alpha": 0.1},
    )


def test_from_json_accepts_string_frequency(
    tmp_path: Path,
) -> None:
    """Test that frequency values as strings are supported."""
    config_path = tmp_path / "config_string_frequency.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "GBR",
                "frequency": "15",
            },
        ),
        encoding="utf-8",
    )

    cfg = SpotOptConfig.from_json(config_path)

    assert cfg == SpotOptConfig(
        model_name=ModelName("GBR"),
        frequency=Frequency(15),
    )


def test_from_json_invalid_combination(tmp_path: Path) -> None:
    """Test error for invalid hyperparameter combination."""
    config_path = tmp_path / "config_invalid.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "Lasso",
                "frequency": 60,
                "mdl_kwargs": {"alpha": 0.1},
                "run_hyperparam_search": True,
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ParameterCombinationError,
        match="Choose either model kwargs",
    ):
        SpotOptConfig.from_json(config_path)


def test_from_dict() -> None:
    """Test that from_dict returns an equivalent SpotOptConfig."""
    cfg = SpotOptConfig.from_dict(
        {
            "model_name": "Lasso",
            "frequency": 60,
            "mdl_kwargs": {"alpha": 0.1},
        },
    )

    assert cfg == SpotOptConfig(
        model_name=ModelName("Lasso"),
        frequency=Frequency(60),
        mdl_kwargs={"alpha": 0.1},
    )


def test_from_dict_accepts_string_frequency() -> None:
    """Test that from_dict supports string frequencies."""
    cfg = SpotOptConfig.from_dict(
        {
            "model_name": "GBR",
            "frequency": "15",
        },
    )

    assert cfg == SpotOptConfig(
        model_name=ModelName("GBR"),
        frequency=Frequency(15),
    )


def test_from_dict_invalid_combination() -> None:
    """Test from_dict error on invalid combination."""
    with pytest.raises(
        ParameterCombinationError,
        match="Choose either model kwargs",
    ):
        SpotOptConfig.from_dict(
            {
                "model_name": "Lasso",
                "frequency": 60,
                "mdl_kwargs": {"alpha": 0.1},
                "run_hyperparam_search": True,
            },
        )
