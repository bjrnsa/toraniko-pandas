# %%"""Complete implementation of the factor model."""

import numpy as np
import pandas as pd

from toraniko_pandas.math import winsorize


def _factor_returns(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    sector_scores: np.ndarray,
    style_scores: np.ndarray,
    residualize_styles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate market, sector, style and residual asset returns for one time period, robust to rank deficiency.

    Parameters
    ----------
    returns: np.ndarray
        Returns of the assets (shape n_assets x 1).
    mkt_caps: np.ndarray
        Asset market capitalizations (shape n_assets x 1).
    sector_scores: np.ndarray
        Asset scores used to estimate the sector return (shape n_assets x m_sectors).
    style_scores: np.ndarray
        Asset scores used to estimate style factor returns (shape n_assets x m_styles).
    residualize_styles: bool
        Indicates if styles should be orthogonalized to market + sector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays: (market/sector/style factor returns, residual returns).
    """
    n_assets = returns.shape[0]
    m_sectors, m_styles = sector_scores.shape[1], style_scores.shape[1]

    # Proxy for the inverse of asset idiosyncratic variances
    W = np.diag(np.sqrt(mkt_caps.ravel()))

    # Estimate sector factor returns with a constraint that the sector factors sum to 0
    # Economically, we assert that the market return is completely spanned by the sector returns
    beta_sector = np.hstack([np.ones(n_assets).reshape(-1, 1), sector_scores])
    a = np.concatenate([np.array([0]), (-1 * np.ones(m_sectors - 1))])
    Imat = np.identity(m_sectors)
    R_sector = np.vstack([Imat, a])

    # Change of variables to add the constraint
    B_sector = beta_sector @ R_sector
    V_sector, _, _, _ = np.linalg.lstsq(
        B_sector.T @ W @ B_sector, B_sector.T @ W, rcond=None
    )

    # Change of variables to recover all sectors
    g = V_sector @ returns
    fac_ret_sector = R_sector @ g
    sector_resid_returns = returns - (B_sector @ g)

    # Estimate style factor returns without constraints
    V_style, _, _, _ = np.linalg.lstsq(
        style_scores.T @ W @ style_scores, style_scores.T @ W, rcond=None
    )
    if residualize_styles:
        fac_ret_style = V_style @ sector_resid_returns
    else:
        fac_ret_style = V_style @ returns

    # Combine factor returns
    fac_ret = np.concatenate([fac_ret_sector, fac_ret_style])

    # Calculate final residuals
    epsilon = sector_resid_returns - (style_scores @ fac_ret_style)

    return fac_ret, epsilon


def estimate_factor_returns(
    returns_df: pd.DataFrame,
    mkt_cap_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    style_df: pd.DataFrame,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate factor and residual returns across all time periods using input asset factor scores.

    Parameters
    ----------
    returns_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | asset_returns |
    mkt_cap_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | market_cap |
    sector_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | followed by one column for each sector.
    style_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | followed by one column for each style.
    winsor_factor: float, optional
        Winsorization proportion (default is 0.05).
    residualize_styles: bool, optional
        Indicates if style returns should be orthogonalized to market + sector returns (default is True).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of Pandas DataFrames melted by date: (factor returns, residual returns).

    Raises
    ------
    TypeError
        If input DataFrames are not valid Pandas DataFrames.
    ValueError
        If required columns are missing from input DataFrames.
    """
    returns, residuals = [], []
    try:
        # Get sector and style columns
        sectors = sorted([col for col in sector_df.columns if col not in ["symbol"]])
        styles = sorted([col for col in style_df.columns if col not in ["symbol"]])

        # Merge all dataframes
        returns_df = (
            returns_df.merge(mkt_cap_df, on=["date", "symbol"])
            .merge(sector_df, on=["date", "symbol"])
            .merge(style_df, on=["date", "symbol"])
        )

        dates = returns_df.index.unique()

        # Process each date
        for _, ddf in returns_df.groupby("date"):
            r = ddf["asset_returns"].to_numpy()

            if winsor_factor is not None:
                r = winsorize(r, winsor_factor)

            f, e = _factor_returns(
                r,
                ddf["market_cap"].to_numpy(),
                ddf[sectors].to_numpy(),
                ddf[styles].to_numpy(),
                residualize_styles,
            )
            returns.append(f)
            residuals.append(dict(zip(ddf["symbol"].tolist(), e)))

    except AttributeError as e:
        raise TypeError("Input DataFrames must be valid Pandas DataFrames") from e

    except KeyError as e:
        raise ValueError(
            "returns_df must have 'date', 'symbol' and 'asset_returns'; "
            "mkt_cap_df must have 'date', 'symbol' and 'market_cap' columns"
        ) from e

    # Create output DataFrames
    ret_df = pd.DataFrame(returns, columns=["market"] + sectors + styles, index=dates)
    eps_df = pd.DataFrame(residuals, index=dates)

    return ret_df, eps_df


# %%
if __name__ == "__main__":
    pass
