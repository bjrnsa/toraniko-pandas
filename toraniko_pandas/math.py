# %% """Basic mathematical and statistical operations used in the model."""

import numpy as np
import pandas as pd


def center_xsection(
    df: pd.DataFrame, target_col: str, over_col: str, standardize: bool = False
) -> pd.Series:
    """Cross-sectionally center (and optionally standardize) a pandas DataFrame column.

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing the data.
    target_col: str
        The column to be standardized.
    over_col: str
        The column over which standardization should be applied, cross-sectionally.
    standardize: bool, optional
        Boolean indicating if we should also standardize the target column (default is False).

    Returns
    -------
    pd.Series
        The centered and optionally standardized column.
    """
    grouped = df.groupby(over_col, observed=True)[target_col]
    mean = grouped.transform("mean")
    centered = df[target_col] - mean

    if standardize:
        std = grouped.transform("std")
        return centered / std
    return centered


def norm_xsection(
    df: pd.DataFrame,
    target_col: str,
    over_col: str,
    lower: int | float = 0,
    upper: int | float = 1,
) -> pd.Series:
    """Cross-sectionally normalize a pandas DataFrame column with rescaling.

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing the data.
    target_col: str
        The column to be normalized.
    over_col: str
        The column over which normalization should be applied, cross-sectionally.
    lower: int or float, optional
        The lower bound of the rescaling (default is 0).
    upper: int or float, optional
        The upper bound of the rescaling (default is 1).

    Returns
    -------
    pd.Series
        The normalized column.
    """
    grouped = df.groupby(over_col, observed=True)[target_col]
    min_val = grouped.transform("min")
    max_val = grouped.transform("max")

    # Preserve NaN and handle min == max case
    mask = max_val != min_val
    result = pd.Series(lower, index=df.index, dtype=float)
    norm = (df[target_col] - min_val) / (max_val - min_val) * (upper - lower) + lower
    result[mask] = norm[mask]
    result[df[target_col].isna()] = np.nan

    return result


def winsorize(data: np.ndarray, percentile: float = 0.05, axis: int = 0) -> np.ndarray:
    """Winsorize each vector of a 2D numpy array to symmetric percentiles.

    Parameters
    ----------
    data: np.ndarray
        The data to be winsorized.
    percentile: float, optional
        The percentiles to apply winsorization at (default is 0.05).
    axis: int, optional
        The axis to apply winsorization over (i.e., orientation if `data` is 2D) (default is 0).

    Returns
    -------
    np.ndarray
        The winsorized data.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not 0 <= percentile <= 1:
        raise ValueError("`percentile` must be between 0 and 1")

    # Return early if all values are NaN
    if np.all(~np.isfinite(data)):
        return data

    fin_data = np.where(np.isfinite(data), data, np.nan)
    lower_bounds = np.nanpercentile(
        fin_data, percentile * 100, axis=axis, keepdims=True
    )
    upper_bounds = np.nanpercentile(
        fin_data, (1 - percentile) * 100, axis=axis, keepdims=True
    )

    return np.clip(data, lower_bounds, upper_bounds)


def winsorize_xsection(
    df: pd.DataFrame,
    data_cols: tuple[str, ...],
    group_col: str,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """Cross-sectionally winsorize DataFrame columns grouped by group_col.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame containing feature data to winsorize.
    data_cols: tuple[str, ...]
        Collection of strings indicating the columns of `df` to be winsorized.
    group_col: str
        Column of `df` to use as the cross-sectional group.
    percentile: float, optional
        Value indicating the symmetric winsorization threshold (default is 0.05).

    Returns
    -------
    pd.DataFrame
        The winsorized DataFrame.
    """
    result = df.copy()

    result[data_cols] = result.groupby(group_col, observed=True, group_keys=False)[
        data_cols
    ].transform(winsorize, percentile=percentile)

    return result


def percentiles_xsection(
    df: pd.DataFrame,
    target_col: str,
    over_col: str,
    lower_pct: float,
    upper_pct: float,
    fill_val: float | int = 0.0,
) -> pd.Series:
    """Cross-sectionally mark values outside percentile thresholds.

    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing the data.
    target_col: str
        Column name to have non-percentile thresholded values masked.
    over_col: str
        Column name to apply masking over, cross-sectionally.
    lower_pct: float
        Lower percentile under which to keep values.
    upper_pct: float
        Upper percentile over which to keep values.
    fill_val: float or int, optional
        Numeric value for masking (default is 0.0).

    Returns
    -------
    pd.Series
        The masked column.
    """

    q_low = df.groupby(over_col)[target_col].transform(
        lambda x: x.quantile(lower_pct, interpolation="lower")
    )
    q_high = df.groupby(over_col)[target_col].transform(
        lambda x: x.quantile(upper_pct, interpolation="higher")
    )

    mask = (df[target_col] <= q_low) | (df[target_col] >= q_high)
    result = df.assign(**{target_col: lambda d: d[target_col].where(mask, fill_val)})

    return result[target_col]


def exp_weights(window: int, half_life: int) -> np.ndarray:
    """Generate exponentially decaying weights over `window` trailing values, decaying by half each `half_life` index.

    Parameters
    ----------
    window: int
        Number of points in the trailing lookback period.
    half_life: int
        Decay rate.

    Returns
    -------
    np.ndarray
        The exponentially decaying weights.
    """
    try:
        assert isinstance(window, int)
        if not window > 0:
            raise ValueError("`window` must be a strictly positive integer")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`window` must be an integer type") from e
    try:
        assert isinstance(half_life, int)
        if not half_life > 0:
            raise ValueError("`half_life` must be a strictly positive integer")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`half_life` must be an integer type") from e

    decay = np.log(2) / half_life
    return np.exp(-decay * np.arange(window))[::-1]


# %%
if __name__ == "__main__":
    pass
