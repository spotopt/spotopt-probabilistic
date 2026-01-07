"""Tests for function check_min_training_data_length."""

from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd
import pytest

from spotopt._constants import MIN_NR_DAYS_TRAIN
from spotopt._exceptions import ShortTrainingDataError
from spotopt._types import Frequency
from spotopt._validation import check_min_training_data_length


@contextmanager
def does_not_raise() -> Generator:
    """Utility context manager signalling no exception."""
    yield


@pytest.mark.parametrize(
    ("frequency", "n_days", "expectation"),
    [
        (Frequency(60), MIN_NR_DAYS_TRAIN, does_not_raise()),
        (Frequency(15), MIN_NR_DAYS_TRAIN, does_not_raise()),
        (
            Frequency(60),
            MIN_NR_DAYS_TRAIN - 1,
            pytest.raises(
                ShortTrainingDataError,
                match=(
                    "The training data should contain more than "
                    f"{MIN_NR_DAYS_TRAIN} days."
                ),
            ),
        ),
        (
            Frequency(15),
            MIN_NR_DAYS_TRAIN - 1,
            pytest.raises(
                ShortTrainingDataError,
                match=(
                    "The training data should contain more than "
                    f"{MIN_NR_DAYS_TRAIN} days."
                ),
            ),
        ),
    ],
)
def test_check_min_training_data_length(
    frequency: Frequency,
    n_days: int,
    expectation,
) -> None:
    """Ensure minimum training period validation behaves as expected."""
    rows_per_day = int(24 * 60 / frequency.value)
    n_rows = n_days * rows_per_day
    df_in = pd.DataFrame(
        data={
            "obs": range(n_rows),
            "fcast": range(n_rows),
        },
    )
    with expectation:
        check_min_training_data_length(df_in, frequency)
