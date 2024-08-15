# Description: Tests for the Heston model for vanilla options.

import numpy as np
import pytest
from pytest import approx

from finmc.utils.assets import Discounter


def test_discounts():
    """Test the Discounter class."""

    # define a discount curve using the zero rate schema
    times = np.array([0.0, 1.0, 2.0, 5.0])
    zero_rates = np.array([0.04, 0.04, 0.045, 0.05])

    test_times = [0.1, 1.0, 3.0, 5.0]
    expected_logdf = [
        0.1 * 0.04,
        1.0 * 0.04,
        2.0 * 0.045 + (0.05 * 5 - 0.045 * 2) * (3 - 2) / (5 - 2),
        5.0 * 0.05,
    ]

    # define a discount curve using the zero rate schema
    zero_data = ("ZERO_RATES", np.column_stack((times, zero_rates)))

    # define a discount curve using the log discount schema
    log_dfs = -times * zero_rates
    logdf_data = ("LOG_DISCOUNTS", np.column_stack((times, log_dfs)))

    for df_data in [zero_data, logdf_data]:
        discounter = Discounter(df_data)
        schema_name = df_data[0]
        print(f"Testing {schema_name} schema")
        for time, logdf in zip(test_times, expected_logdf):
            df = discounter.discount(time)
            expected_df = np.exp(-logdf)
            assert df == approx(expected_df, abs=1e-6)
            print(f"{df:11.6f} {expected_df:11.6f} {df - expected_df:9.6f}")


if __name__ == "__main__":
    pytest.main([__file__])
