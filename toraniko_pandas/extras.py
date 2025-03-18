import cvxpy as cp
import numpy as np
import pandas as pd

from toraniko_pandas.math import winsorize


def _factor_returns_cvxpy(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    scores: np.ndarray,
    group_sizes: list[int],
    residualize_styles: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate factor returns using CVXPY optimization with sum-zero constraints.

    Parameters
    ----------
    returns : np.ndarray
        Returns of the assets (shape n_assets x 1)
    mkt_caps : np.ndarray
        Asset market capitalizations (shape n_assets x 1)
    scores : np.ndarray
        Matrix of factor exposures (categorical and style), shape (n_assets x n_factors)
    group_sizes : list[int]
        List containing the number of factors in each sum-zero group.
        e.g., [11, 24] for 11 sectors and 24 industries
    residualize_styles : bool, optional
        Whether to residualize style factors against sum-zero factors, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (factor returns, residual returns)
    """

    # Calculate split points for constrained factors
    split_points = np.cumsum([0] + group_sizes)
    n_constrained = split_points[-1]

    # Split data into constrained and unconstrained factors
    X_const = scores[:, :n_constrained]
    X_style = scores[:, n_constrained:]

    # Create CVXPY variables
    const_returns = cp.Variable(n_constrained)
    style_returns = cp.Variable(X_style.shape[1]) if X_style.shape[1] > 0 else None

    # Create sum-zero constraints for each group
    constraints = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        constraints.append(cp.sum(const_returns[start:end]) == 0)

    # Define predicted returns
    if residualize_styles and style_returns is not None:
        # First fit constrained factors
        predicted_returns_const = X_const @ const_returns
        objective_const = cp.Minimize(
            cp.sum_squares(
                cp.multiply(np.sqrt(mkt_caps), returns - predicted_returns_const)
            )
        )
        problem_const = cp.Problem(objective_const, constraints)
        problem_const.solve()

        # Then fit styles on residuals
        residuals = returns - X_const @ const_returns.value
        predicted_returns_style = X_style @ style_returns
        objective_style = cp.Minimize(
            cp.sum_squares(
                cp.multiply(np.sqrt(mkt_caps), residuals - predicted_returns_style)
            )
        )
        problem_style = cp.Problem(objective_style, [])
        problem_style.solve()

        # Combine factor returns
        factor_returns = np.concatenate([const_returns.value, style_returns.value])

    else:
        # Fit all factors simultaneously
        if style_returns is not None:
            predicted_returns = X_const @ const_returns + X_style @ style_returns
            objective = cp.Minimize(
                cp.sum_squares(
                    cp.multiply(np.sqrt(mkt_caps), returns - predicted_returns)
                )
            )
            problem = cp.Problem(objective, constraints)
            problem.solve()
            factor_returns = np.concatenate([const_returns.value, style_returns.value])
        else:
            predicted_returns = X_const @ const_returns
            objective = cp.Minimize(
                cp.sum_squares(
                    cp.multiply(np.sqrt(mkt_caps), returns - predicted_returns)
                )
            )
            problem = cp.Problem(objective, constraints)
            problem.solve()
            factor_returns = const_returns.value

    # Calculate residuals
    predicted = scores @ factor_returns
    epsilon = returns - predicted

    return factor_returns, epsilon


def estimate_factor_returns_cvxpy(
    returns_df: pd.DataFrame,
    mkt_cap_df: pd.DataFrame,
    factor_dfs: list[pd.DataFrame],
    factor_groups: list[int],
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate factor and residual returns across all time periods using CVXPY optimization with sum-zero constraints.

    Parameters
    ----------
    returns_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | asset_returns |
    mkt_cap_df: pd.DataFrame
        Pandas DataFrame containing columns: | date | symbol | market_cap |
    factor_dfs: list[pd.DataFrame]
        List of DataFrames, each containing: | date | symbol | followed by factor columns
        The order should match factor_groups (constrained factors first, then unconstrained)
    factor_groups: list[int]
        List specifying the number of factors in each constrained group
        (e.g., [11, 24] for 11 sectors and 24 industries)
    winsor_factor: float, optional
        Winsorization proportion (default is 0.05)
    residualize_styles: bool, optional
        Indicates if style returns should be orthogonalized to constrained factors (default is True)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of Pandas DataFrames: (factor returns, residual returns)

    Raises
    ------
    TypeError
        If input DataFrames are not valid Pandas DataFrames
    ValueError
        If required columns are missing from input DataFrames
    """
    fac_returns, residuals = [], []
    try:
        # Create a combined DataFrames
        merged_df = returns_df.merge(mkt_cap_df, on=["date", "symbol"])

        # Add each factor DataFrame
        all_factor_cols = []
        for factor_df in factor_dfs:
            factor_cols = [
                col for col in factor_df.columns if col not in ["date", "symbol"]
            ]
            all_factor_cols.extend(factor_cols)
            merged_df = merged_df.merge(factor_df, on=["date", "symbol"])

        dates = merged_df.index.unique()

        # Process each date
        for _, ddf in merged_df.groupby("date"):
            r = ddf["asset_returns"].to_numpy()

            if winsor_factor is not None:
                r = winsorize(r, winsor_factor)

            mkt_caps = ddf["market_cap"].to_numpy()

            # Create the factor scores matrix
            scores = ddf[all_factor_cols].to_numpy()

            # Run the CVXPY optimization
            f, e = _factor_returns_cvxpy(
                r,
                mkt_caps,
                scores,
                factor_groups,
                residualize_styles,
            )

            fac_returns.append(f)
            residuals.append(dict(zip(ddf["symbol"].tolist(), e)))

    except AttributeError as e:
        raise TypeError("Input DataFrames must be valid Pandas DataFrames") from e

    except KeyError as e:
        raise ValueError(
            "returns_df must have 'date', 'symbol' and 'asset_returns'; "
            "mkt_cap_df must have 'date', 'symbol' and 'market_cap' columns"
        ) from e

    # Create output DataFrames
    ret_df = pd.DataFrame(fac_returns, columns=all_factor_cols, index=dates)
    eps_df = pd.DataFrame(residuals, index=dates)

    return ret_df, eps_df


def shrinkage_cov(
    returns: np.ndarray, return_all: bool = False
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """Shrinks sample covariance matrix towards constant correlation unequal variance matrix.

    Implements Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management,
    30(2004), 110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1
    (constant sample average correlation unequal sample variance matrix).

    Parameters
    ----------
    returns : np.ndarray
        Matrix of returns with shape (t, n) where t is the number of observations
        and n is the number of assets.
    return_all : bool, optional
        If True, returns a tuple of (sigma, average_cor, shrink), otherwise returns only sigma.
        Default is False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, float, float]
        If return_all is False (default): Shrunk covariance matrix only
        If return_all is True: Tuple containing:
            - Shrunk covariance matrix
            - Sample average correlation
            - Shrinkage intensity (between 0 and 1)
    """
    t, n = returns.shape
    mean_returns = np.mean(returns, axis=0, keepdims=True)
    demeaned_returns = returns - mean_returns
    sample_cov = demeaned_returns.T @ demeaned_returns / t

    # Sample average correlation
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = np.sqrt(var)
    unit_cor_var = sqrt_var @ sqrt_var.T
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / (n * (n - 1))
    prior = average_cor * unit_cor_var
    np.fill_diagonal(prior, var)

    # Pi-hat
    y = demeaned_returns**2
    phi_mat = (y.T @ y) / t - sample_cov**2
    phi = phi_mat.sum()

    # Rho-hat
    theta_mat = ((demeaned_returns**3).T @ demeaned_returns) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = (
        np.diag(phi_mat).sum()
        + average_cor * (1 / (sqrt_var @ sqrt_var.T) * theta_mat).sum()
    )

    # Gamma-hat
    gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

    # Shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # Estimator
    shrink_cov = shrink * prior + (1 - shrink) * sample_cov

    if return_all:
        return shrink_cov, average_cor, shrink
    return shrink_cov


def ewma_cov(eps: np.ndarray | pd.DataFrame, half_life: int) -> np.ndarray:
    """
    Calculate the EWMA empirical covariance matrix. Commonly used for
    calculating the covariance matrix of the residuals from a factor model.

    Parameters:
    -----------
    eps : pandas.DataFrame
        DataFrame of estimated idiosyncratic returns, with dates as index
        and assets as columns
    half_life : int
        Half-life parameter for exponential weighting

    Returns
    --------
    idiosyncratic_cov : numpy.ndarray
        EWMA empirical idiosyncratic covariance matrix
    """
    # Convert to numpy array for calculations
    if isinstance(eps, pd.DataFrame):
        epsilon = np.array(eps)
    else:
        epsilon = eps
    T = len(epsilon)

    # Create exponential weights
    weights = np.exp(-np.arange(T - 1, -1, -1) / half_life)

    # Normalize weights to sum to T
    kappa = T / np.sum(weights)
    weights = kappa * weights

    # Create diagonal weighting matrix
    W = np.diag(weights)

    # Calculate EWMA empirical idiosyncratic covariance matrix
    idiosyncratic_cov = epsilon.T @ W @ epsilon

    return idiosyncratic_cov


def autocorr_cov(C, lags):
    """
    Implements the first estimator from the document:
    Ω_f = C₀ + 1/2 ∑(Cₗ + Cₗ')

    Parameters:
    -----------
    C : list of numpy.ndarray
        List of covariance matrices where C[0] is C₀, C[1] is C₁, etc.
    lags : int
        Number of lags to include

    Returns:
    --------
    Omega_f : numpy.ndarray
        Autocorrelation-consistent estimator of factor covariance
    """
    C0 = C[0]
    correction = np.zeros_like(C0)

    for lag in range(1, lags + 1):
        Cl = C[lag]
        correction += Cl + Cl.T

    Omega_f = C0 + 0.5 * correction
    return Omega_f


def nwest_cov(C, lags):
    """
    Implements Newey and West's estimator:
    Ω_f = C₀ + ∑(1 - l/(1+lmax))(Cₗ + Cₗ')

    Parameters:
    -----------
    C : list of numpy.ndarray
        List of covariance matrices where C[0] is C₀, C[1] is C₁, etc.
    lags : int
        Number of lags to include (lmax)

    Returns:
    --------
    Omega_f : numpy.ndarray
        Newey-West estimator of factor covariance
    """
    C0 = C[0]
    correction = np.zeros_like(C0)

    for lag in range(1, lags + 1):
        Cl = C[lag]
        weight = 1 - lag / (1 + lags)
        correction += weight * (Cl + Cl.T)

    Omega_f = C0 + correction
    return Omega_f


def lagged_covs(factor_returns, max_lag, method: str = "shrinkage"):
    """
    Calculate lagged covariance matrices

    Parameters:
    -----------
    factor_returns : numpy.ndarray
        Matrix of factor returns, shape (T, N)
    max_lag : int
        Maximum lag to calculate

    Returns:
    --------
    C : list of numpy.ndarray
        List of covariance matrices [C₀, C₁, ..., C_max_lag]
    """
    T, N = factor_returns.shape
    C = []

    # Calculate C₀ (contemporaneous covariance)
    if method == "shrinkage":
        C0 = shrinkage_cov(factor_returns)
    elif method == "empirical":
        C0 = np.cov(factor_returns, rowvar=False)
    else:
        raise ValueError(f"Invalid method: {method}")
    C.append(C0)

    # Calculate lagged covariances
    for lag in range(1, max_lag + 1):
        returns_t = factor_returns[lag:, :]
        returns_t_minus_l = factor_returns[:-lag, :]

        # [C_l]i,j = cov(f_i,t, f_j,t-l)
        Cl = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Cl[i, j] = np.cov(returns_t[:, i], returns_t_minus_l[:, j])[0, 1]

        C.append(Cl)

    return C
