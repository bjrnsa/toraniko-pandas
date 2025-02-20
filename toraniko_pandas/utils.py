# %%  """Utility functions, primarily for data cleaning."""

import numpy as np
import pandas as pd


def fill_features(
    df: pd.DataFrame,
    features: tuple[str, ...],
    over_col: str,
) -> pd.DataFrame:
    """
    Cast feature columns to numeric, handle NaN/inf values, and forward fill nulls.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    features (tuple[str, ...]): A tuple of feature column names to be processed.
    over_col (str): The column name to group by for forward filling.

    Returns:
    pd.DataFrame: The processed DataFrame with features cast to numeric, NaN/inf values handled, and forward filled within groups.

    Raises:
    ValueError: If the DataFrame does not contain all required columns.
    """
    try:
        assert all(c in df.columns for c in features + (over_col,))

        result = df.copy()
        result = result.sort_index()

        for feature in features:
            # Cast to float and handle special values
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
            result[feature] = result[feature].replace([np.inf, -np.inf], np.nan)

            # Forward fill within groups
            result[feature] = result.groupby(over_col, observed=True)[
                feature
            ].transform(lambda x: x.ffill())

        return result
    except AssertionError:
        raise ValueError(
            f"`df` must have all of {[over_col] + list(features)} as columns"
        )


def smooth_features(
    df: pd.DataFrame,
    features: tuple[str, ...],
    over_col: str,
    window_size: int,
) -> pd.DataFrame:
    """
    Smooth features using rolling mean within groups.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    features (tuple[str, ...]): A tuple of feature column names to be smoothed.
    over_col (str): The column name to group by for smoothing.
    window_size (int): The window size for the rolling mean.

    Returns:
    pd.DataFrame: The processed DataFrame with smoothed features using rolling mean within groups.

    Raises:
    ValueError: If the DataFrame does not contain all required columns.
    """
    try:
        assert all(c in df.columns for c in features + (over_col,))

        result = df.copy()
        result = result.sort_index()

        for feature in features:
            result[feature] = result.groupby(over_col)[feature].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )

        return result
    except AssertionError:
        raise ValueError(
            f"`df` must have all of {[over_col] + list(features)} as columns"
        )


def top_n_by_group(
    df: pd.DataFrame,
    n: int,
    rank_var: str,
    group_var: tuple[str, ...],
    filter: bool = True,
) -> pd.DataFrame:
    """
    Mark or filter top n rows within groups.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    n (int): The number of top rows to select within each group.
    rank_var (str): The column name to rank by.
    group_var (tuple[str, ...]): A tuple of column names to group by.
    filter (bool): If True, filter the DataFrame to only include the top n rows within each group. If False, add a rank mask column.

    Returns:
    pd.DataFrame: The processed DataFrame with top n rows marked or filtered within groups.

    Raises:
    ValueError: If the DataFrame does not contain all required columns.
    """
    try:
        assert all(c in df.columns for c in (rank_var,))

        result = df.copy()
        result["rank"] = (
            result.groupby(list(group_var), observed=True)[rank_var]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        if filter:
            return (
                result[result["rank"] <= n]
                .drop(columns=["rank"])
                .sort_values(by=list(group_var) + [rank_var])
            )
        else:
            result["rank_mask"] = (result["rank"] <= n).astype(int)
            return result.drop(columns=["rank"]).sort_values(
                by=list(group_var) + [rank_var]
            )

    except AssertionError:
        raise ValueError(
            f"`df` is missing one or more required columns: '{rank_var}' and '{group_var}'"
        )


# %%
if __name__ == "__main__":
    pass
