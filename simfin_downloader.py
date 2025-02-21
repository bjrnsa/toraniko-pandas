import os
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import simfin as sf
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SIMFIN_API_KEY")


def simfin_df(func):
    """
    Decorator to format DataFrame returned by SimFin API functions.

    Parameters
    ----------
    func : function
        The function to be decorated.

    Returns
    -------
    function
        The decorated function that formats the DataFrame.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        df = func(self, *args, **kwargs)
        if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex):
            df.index.names = ["symbol", "date"]
            df = df.reset_index()
            df["symbol"] = df["symbol"].astype("category")
            df = df.set_index(["symbol", "date"])
        return df

    return wrapper


def fill_features_simfin(
    df: pd.DataFrame,
    features: tuple[str, ...],
    over_col: str,
) -> pd.DataFrame:
    """
    Cast feature columns to numeric, handle NaN/inf values, and forward fill nulls.
    Handles MultiIndex DataFrames outputted by SimFin API.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    features : tuple[str, ...]
        A tuple of feature column names to be processed.
    over_col : str
        The column name to group by for forward filling.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with features cast to numeric, NaN/inf values handled, and forward filled within groups.

    Raises
    ------
    ValueError
        If the DataFrame does not contain all required columns.
    """
    try:
        assert all(c in df.columns for c in features)

        result = df.copy()
        result = result.sort_index()

        for feature in features:
            # Cast to float and handle special values
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
            result[feature] = result[feature].replace([np.inf, -np.inf], np.nan)

            # Forward fill within groups
            result[feature] = result.groupby(over_col)[feature].transform(
                lambda x: x.ffill()
            )

            # Downcast numeric types to reduce memory
            result[feature] = pd.to_numeric(result[feature], downcast="float")

        return result
    except AssertionError:
        raise ValueError(f"`df` must have all of {list(features)} as columns")


class SimFin(sf.StockHub):
    """
    Class to interact with the SimFin API and format data for Toraniko.

    Attributes
    ----------
    variant : str
        The variant of the data (e.g., "daily").
    market : str
        The market to use (e.g., "us").
    firm_info : pd.DataFrame
        DataFrame containing firm information.
    tickers : pd.Series
        Series containing unique tickers.

    Methods
    -------
    get_toraniko_data()
        Get the data in a format that Toraniko can use.
    _load_companies()
        Load company data from SimFin.
    load_returns(stock_prices)
        Load the returns data.
    load_value_data()
        Load the value data.
    get_scores(group_col)
        Get sector or industry scores.
    format_company_data()
        Load the sector data.
    _load_shareprices(variant)
        Load share prices from SimFin.
    _val_signals(variant)
        Load value signals from SimFin.
    _price_signals(variant)
        Load price signals from SimFin.
    _volume_signals(variant)
        Load volume signals from SimFin.
    _growth_signals(variant)
        Load growth signals from SimFin.
    _fin_signals(variant)
        Load financial signals from SimFin.
    """

    def __init__(self, market: str = "us", variant: str = "daily"):
        """
        Initialize the SimFin API.

        Parameters
        ----------
        market : str, optional
            The market to use (default is "us").
        variant : str, optional
            The variant of the data (default is "daily").
        """
        super().__init__(market=market)
        sf.set_api_key(API_KEY)
        sf.set_data_dir("~/data/simfin")

        self.variant = variant
        self.market = market  # Storing market as attribute

        self.format_company_data()

    def get_toraniko_data(self):
        """
        Get the data in a format that Toraniko can use.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing the main DataFrame, sector DataFrame, and industry DataFrame.
        """
        # Load dataframes inside the method.
        stock_prices = self._load_shareprices(variant=self.variant)
        returns_df = self.load_returns(stock_prices)
        value_df = self.load_value_data()

        df = returns_df.merge(value_df, on=["symbol", "date"])
        sector_df = self.get_scores("Sector")
        industry_df = self.get_scores("Industry")

        return df, sector_df, industry_df

    def _load_companies(self):
        """
        Load company data from SimFin.

        Returns
        -------
        pd.DataFrame
            DataFrame containing company data.
        """
        # Explicitly define the market
        companies = self.load_companies()
        industries = sf.load_industries()

        df = (
            companies.reset_index()
            .merge(industries, on="IndustryId")[["Ticker", "Sector", "Industry"]]
            .rename(columns={"Ticker": "symbol"})
        )

        # Downcast to reduce memory
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")

        return df

    def load_returns(self, stock_prices):
        """
        Load the returns data.

        Parameters
        ----------
        stock_prices : pd.DataFrame
            DataFrame containing stock prices.

        Returns
        -------
        pd.DataFrame
            DataFrame containing returns data.
        """
        returns_df = (
            stock_prices.groupby("symbol")["Adj. Close"]
            .pct_change()
            .rename("asset_returns")
            .reset_index("symbol")
            .sort_values(["date", "symbol"])
            .dropna()
        )

        # Downcast to reduce memory
        returns_df["asset_returns"] = pd.to_numeric(
            returns_df["asset_returns"], downcast="float"
        )
        return returns_df

    def load_value_data(self):
        """
        Load the value data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing value data.
        """
        new_names = {
            "Price to Book Value": "book_price",
            "P/Cash": "cf_price",
            "P/Sales": "sales_price",
            "Market-Cap": "market_cap",
        }
        value_df = self._val_signals(variant=self.variant)
        value_df = (1 / value_df[new_names.keys()]).rename(columns=new_names)
        value_df = fill_features_simfin(value_df, list(new_names.values()), "symbol")
        value_df = value_df.assign(market_cap=1 / value_df["market_cap"])
        value_df = value_df.astype({col: "float32" for col in new_names.values()})

        return value_df.reset_index("symbol")

    def get_scores(self, group_col: str):
        """
        Get sector or industry scores.

        Parameters
        ----------
        group_col : str
            The column name to group by (e.g., "Sector" or "Industry").

        Returns
        -------
        pd.DataFrame
            DataFrame containing scores.
        """
        scores_df = pd.get_dummies(
            self.firm_info[group_col], dtype="int8", drop_first=False, prefix=group_col
        ).set_index(self.tickers)
        scores_df.columns = (
            scores_df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")
        )
        return scores_df

    def format_company_data(self):
        """
        Load the sector data.
        """
        self.firm_info = self._load_companies()
        self.tickers = pd.Series(self.firm_info["symbol"].unique(), name="symbol")

    @simfin_df
    def _load_shareprices(self, variant: str = "daily"):
        """
        Load share prices from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing share prices.
        """
        return self.load_shareprices(variant=variant)

    @simfin_df
    def _val_signals(self, variant: str = "daily"):
        """
        Load value signals from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing value signals.
        """
        return self.val_signals(variant=variant)

    @simfin_df
    def _price_signals(self, variant: str = "daily"):
        """
        Load price signals from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing price signals.
        """
        return self.price_signals(variant=variant)

    @simfin_df
    def _volume_signals(self, variant: str = "daily"):
        """
        Load volume signals from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing volume signals.
        """
        return self.volume_signals(variant=variant)

    @simfin_df
    def _growth_signals(self, variant: str = "daily"):
        """
        Load growth signals from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing growth signals.
        """
        return self.growth_signals(variant=variant)

    @simfin_df
    def _fin_signals(self, variant: str = "daily"):
        """
        Load financial signals from SimFin.

        Parameters
        ----------
        variant : str
            The variant of the data (e.g., "daily").

        Returns
        -------
        pd.DataFrame
            DataFrame containing financial signals.
        """
        return self.fin_signals(variant=variant)


if __name__ == "__main__":
    # Disable FutureWarnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Ignore warning when logs are negative or divide by zero
    warnings.filterwarnings("ignore", message="invalid value encountered in log")
    warnings.filterwarnings("ignore", message="divide by zero encountered in log")

    data = SimFin()
    df, sector_df, industry_df = data.get_toraniko_data()
