# %% Import Dependencies
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from simfin_downloader import SimFin
from toraniko_pandas.extras import (
    autocorr_cov,
    estimate_factor_returns_cvxpy,
    ewma_cov,
    lagged_covs,
    nwest_cov,
    shrinkage_cov,
)
from toraniko_pandas.model import estimate_factor_returns
from toraniko_pandas.styles import factor_mom, factor_sze, factor_val
from toraniko_pandas.utils import top_n_by_group

# Configure warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

# %% Load SimFin Data
simfin = SimFin()
df, sectors, industries = simfin.get_toraniko_data()

# Filter top 3000 companies by market cap
df = top_n_by_group(df.dropna(), 3000, "market_cap", ("date",), True)


# %% Calculate Market Factors
# Size factor calculation
size_df = factor_sze(df, lower_decile=0.2, upper_decile=0.8)

# Momentum factor calculation
mom_df = factor_mom(df, trailing_days=252, winsor_factor=0.01)

# Value factor calculation
value_df = factor_val(df, winsor_factor=0.05)

# %%
# Momentum distribution
mom_df["mom_score"].hist(bins=100)
plt.title("Momentum Factor Distribution")
plt.show()

# Value distribution
value_df["val_score"].hist(bins=100)
plt.title("Value Factor Distribution")
plt.show()


# %% Merge Data Sources
style_scores = (
    value_df.merge(mom_df, on=["symbol", "date"])
    .merge(size_df, on=["symbol", "date"])
    .dropna()
)

ddf = (
    df.reset_index()[["date", "symbol", "asset_returns", "market_cap"]]
    .merge(sectors, on="symbol")
    .merge(industries, on="symbol")
    .merge(style_scores, on=["symbol", "date"])
    .dropna()
    .astype({"symbol": "category"})
    .set_index("date")
    .sort_index()
)


# %% Calculate Factor Returns
CVXPY = True

returns_df = ddf[["symbol", "asset_returns"]]
mkt_cap_df = ddf[["symbol", "market_cap"]]
sector_df = ddf[sectors.columns.tolist() + ["symbol"]]
industry_df = ddf[industries.columns.tolist() + ["symbol"]]
style_df = ddf[style_scores.columns]
n_industries = len(industries.columns.tolist())
n_sectors = len(sectors.columns.tolist())


if CVXPY:
    fac_df, eps_df = estimate_factor_returns_cvxpy(
        returns_df,
        mkt_cap_df,
        [sector_df, industry_df, style_df],
        [n_sectors, n_industries],
        winsor_factor=0.1,
        residualize_styles=False,
    )
else:
    fac_df, eps_df = estimate_factor_returns(
        returns_df,
        mkt_cap_df,
        sector_df,
        style_df,
        winsor_factor=0.1,
        residualize_styles=False,
    )


# %% Test Factor Model
random_symbol = np.random.choice(returns_df["symbol"].unique())
symbol_sector = sectors.loc[random_symbol].idxmax()
symbol_industry = industries.loc[random_symbol].idxmax()

y = returns_df.query("symbol == @random_symbol")["asset_returns"]
X = fac_df[
    [
        symbol_sector,
        symbol_industry,
        "mom_score",
        "val_score",
        "sze_score",
    ]
]
X = sm.add_constant(X).loc[y.index]

model = sm.OLS(y, X)
results = model.fit().get_robustcov_results()
results.summary()


# %% Factor Covariance Matrix
returns_pivot = returns_df.pivot(columns="symbol", values="asset_returns").dropna(
    axis=1
)
tickers = returns_pivot.columns
X = returns_pivot.to_numpy()
F = fac_df.to_numpy()

B = np.linalg.lstsq(F, X, rcond=None)[0]
C_f = shrinkage_cov(F, False)
V = np.var(X, axis=0)

Cs = lagged_covs(F, 3, method="shrinkage")
C_newey = nwest_cov(Cs, 3)
C_autocorr = autocorr_cov(Cs, 3)

C_x_shrunk = B.T @ C_f @ B
C_x_newey = B.T @ C_newey @ B
C_x_autocorr = B.T @ C_autocorr @ B

factor_cov = pd.DataFrame(C_x_autocorr, index=tickers, columns=tickers)

# %% Idiosyncratic Covariance Matrix

V = ewma_cov(eps_df[tickers], 121)
idiosyncratic_cov = pd.DataFrame(V, index=tickers, columns=tickers)
