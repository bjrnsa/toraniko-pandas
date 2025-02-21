# %%"""Style factor implementations."""

import numpy as np
import pandas as pd

from toraniko_pandas.math import (
    center_xsection,
    exp_weights,
    percentiles_xsection,
    winsorize_xsection,
)

### This bears repeating from the original implementation:

###
# NB: These functions do not try to handle NaN or null resilience for you, nor do they make allowances
# for data having pathological distributions. Garbage in, garbage out. You need to inspect your data
# and use the functions in the math and utils modules to ensure your features are sane and
# well-behaved before you try to construct factors from them!
###


def factor_mom(
    returns_df: pd.DataFrame,
    trailing_days: int = 504,
    half_life: int = 126,
    lag: int = 20,
    winsor_factor: float = 0.01,
) -> pd.DataFrame:
    """Estimate rolling symbol by symbol momentum factor scores using asset returns.

    Parameters
    ----------
    returns_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | asset_returns |
    trailing_days: int, optional
        Look back period over which to measure momentum (default is 504).
    half_life: int, optional
        Decay rate for exponential weighting, in days (default is 126).
    lag: int, optional
        Number of days to lag current day's return observation (default is 20).
    winsor_factor: float, optional
        Winsorization threshold (default is 0.01).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | mom_score |

    Raises
    ------
    TypeError
        If returns_df is not a Pandas DataFrame with required columns.
    KeyError
        If required columns are missing from returns_df.
    """
    weights = exp_weights(trailing_days, half_life)

    def weighted_cumprod(values: np.ndarray) -> float:
        if isinstance(values, pd.Series):
            values = np.array(values)

        if len(values) < len(weights):
            return np.nan
        return (np.cumprod(1 + (values * weights[-len(values) :])) - 1)[-1]

    try:
        # Sort and create lagged returns
        df = returns_df.sort_index().copy()
        df["asset_returns"] = df.groupby("symbol", observed=True)[
            "asset_returns"
        ].shift(lag)

        # Calculate momentum scores
        df = (
            df.groupby("symbol", group_keys=True, observed=True)["asset_returns"]
            .rolling(window=trailing_days, min_periods=trailing_days)
            .apply(weighted_cumprod)
            .rename("mom_score")
            .reset_index("symbol")
            .sort_index()
        )

        # Winsorize and center
        df = winsorize_xsection(df, ["mom_score"], "date", winsor_factor)
        df["mom_score"] = center_xsection(df, "mom_score", "date", True)

        return df[["symbol", "mom_score"]]

    except AttributeError as e:
        raise TypeError(
            "returns_df must be a Pandas DataFrame with required columns"
        ) from e
    except KeyError as e:
        raise e


def factor_sze(
    mkt_cap_df: pd.DataFrame,
    lower_decile: float = 0.2,
    upper_decile: float = 0.8,
) -> pd.DataFrame:
    """Estimate rolling symbol by symbol size factor scores using asset market caps.

    Parameters
    ----------
    mkt_cap_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | market_cap |
    lower_decile: float, optional
        Lower percentile cutoff for winsorization (default is 0.2).
    upper_decile: float, optional
        Upper percentile cutoff for winsorization (default is 0.8).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | sze_score |

    Raises
    ------
    TypeError
        If mkt_cap_df is not a Pandas DataFrame with required columns.
    ValueError
        If required columns are missing from mkt_cap_df.
    """
    try:
        df = mkt_cap_df.copy()

        # Calculate small-minus-big factor using log market cap
        df["sze_score"] = -1 * np.log(df["market_cap"])

        # Center and standardize within each date
        df["sze_score"] = center_xsection(df, "sze_score", "date", True)

        # Apply percentile cutoffs
        df["sze_score"] = percentiles_xsection(
            df, "sze_score", "date", lower_decile, upper_decile, 0.0
        )

        return df[["symbol", "sze_score"]]

    except AttributeError as e:
        raise TypeError(
            "mkt_cap_df must be a Pandas DataFrame with required columns"
        ) from e
    except KeyError as e:
        raise ValueError(
            "mkt_cap_df must have 'date', 'symbol' and 'market_cap' columns"
        ) from e


def factor_val(
    value_df: pd.DataFrame,
    winsor_factor: float | None = None,
) -> pd.DataFrame:
    """Estimate rolling symbol by symbol value factor scores using price ratios.

    Parameters
    ----------
    value_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | book_price | sales_price | cf_price
    winsor_factor: float, optional
        Winsorization threshold. None if no winsorization is applied (default is None).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing: | date | symbol | val_score |

    Raises
    ------
    TypeError
        If value_df is not a Pandas DataFrame.
    ValueError
        If required columns are missing from value_df.
    """
    try:
        df = value_df.copy()

        # Winsorize if specified
        if winsor_factor is not None:
            df = winsorize_xsection(
                df,
                ["book_price", "sales_price", "cf_price"],
                "date",
                winsor_factor,
            )

        # Log transform price ratios
        df["book_price"] = np.log(df["book_price"])
        df["sales_price"] = np.log(df["sales_price"])

        # Center and standardize features
        df["book_price"] = center_xsection(df, "book_price", "date", True)
        df["sales_price"] = center_xsection(df, "sales_price", "date", True)
        df["cf_price"] = center_xsection(df, "cf_price", "date", True)

        # Calculate value score
        # NB: it's imperative you've properly handled NaNs prior to this point
        df["val_score"] = df[["book_price", "sales_price", "cf_price"]].mean(axis=1)

        # Center and standardize value score
        df["val_score"] = center_xsection(df, "val_score", "date", True)

        return df[["symbol", "val_score"]]

    except AttributeError as e:
        raise TypeError("value_df must be a Pandas DataFrame") from e

    except KeyError as e:
        raise ValueError(
            "value_df must have 'date', 'symbol', 'book_price', 'sales_price' and 'cf_price' columns"
        ) from e


if __name__ == "__main__":
    pass
