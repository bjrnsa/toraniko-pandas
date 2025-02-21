# toraniko-pandas [![PyPI](https://img.shields.io/pypi/v/toraniko-pandas)](https://pypi.org/project/toraniko-pandas/)

Pandas implementation of institutional-grade multi-factor risk model (Polars original by [0xfdf/toraniko](https://github.com/0xfdf/toraniko))

## Installation

```
pip install toraniko_pandas
```

## User Manual

This notebook demonstrates factor analysis using SimFin financial data and custom factor models.

### Important
You need free API key from [SimFin](https://www.simfin.com/en/) to run my example. See their [Github](https://github.com/SimFin/simfin) for more info.
Create a .env file and place it like below
```
# SimFin 
SIMFIN_API_KEY = "YOUR_KEY"
```

### Data
```python
import warnings

import matplotlib.pyplot as plt
import statsmodels.api as sm

from simfin_downloader import SimFin
from toraniko_pandas.model import estimate_factor_returns
from toraniko_pandas.styles import factor_mom, factor_sze, factor_val
from toraniko_pandas.utils import top_n_by_group

# Configure warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in log")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
```

```python
simfin = SimFin()
df, sectors, industries = simfin.get_toraniko_data()

# Filter top 3000 companies by market cap
df = top_n_by_group(df.dropna(), 3000, "market_cap", ("date",), True)
```

```python
# Display sector data
sectors
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sector_basic_materials</th>
      <th>sector_business_services</th>
      <th>sector_consumer_cyclical</th>
      <th>sector_consumer_defensive</th>
      <th>sector_energy</th>
      <th>sector_financial_services</th>
      <th>sector_healthcare</th>
      <th>sector_industrials</th>
      <th>sector_other</th>
      <th>sector_real_estate</th>
      <th>sector_technology</th>
      <th>sector_utilities</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AA</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AAC</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AACI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AAC_delisted</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZWS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZY</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYME</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYNE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYXI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5766 rows × 12 columns</p>
</div>

```python
# Display industry data
industries

```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>industry_advertising_and_marketing_services</th>
      <th>industry_aerospace_and_defense</th>
      <th>industry_agriculture</th>
      <th>industry_airlines</th>
      <th>industry_alternative_energy_sources_and_other</th>
      <th>industry_application_software</th>
      <th>industry_asset_management</th>
      <th>industry_autos</th>
      <th>industry_banks</th>
      <th>industry_beverages_-_alcoholic</th>
      <th>...</th>
      <th>industry_retail_-_defensive</th>
      <th>industry_semiconductors</th>
      <th>industry_steel</th>
      <th>industry_tobacco_products</th>
      <th>industry_transportation_and_logistics</th>
      <th>industry_travel_and_leisure</th>
      <th>industry_truck_manufacturing</th>
      <th>industry_utilities_-_independent_power_producers</th>
      <th>industry_utilities_-_regulated</th>
      <th>industry_waste_management</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AA</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AAC</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AACI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AAC_delisted</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZWS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZY</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYME</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYNE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZYXI</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5766 rows × 74 columns</p>
</div>

```python
# Asset returns was calculated by taking the percentage change of the adjusted close prices
ret_df = df[["symbol", "asset_returns"]]
ret_df
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>asset_returns</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-01</th>
      <td>EOSS</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>YTEN</td>
      <td>0.017544</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>GIDYL</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>VISL</td>
      <td>-0.028571</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>ASLE</td>
      <td>-0.001020</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>NLY</td>
      <td>-0.012099</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>AAPL</td>
      <td>-0.008461</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>MSFT</td>
      <td>-0.006159</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>WPC</td>
      <td>0.001669</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>RENT</td>
      <td>-0.045675</td>
    </tr>
  </tbody>
</table>
<p>3406422 rows × 2 columns</p>
</div>

```python
# Market cap is used to calculate the size factor and later to select the top N by market cap
cap_df = df[["symbol", "market_cap"]]
cap_df
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>market_cap</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-01</th>
      <td>EOSS</td>
      <td>1.600000e+05</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>YTEN</td>
      <td>2.939904e+05</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>GIDYL</td>
      <td>3.113633e+05</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>VISL</td>
      <td>3.423120e+05</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>ASLE</td>
      <td>3.623279e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>NLY</td>
      <td>2.291316e+12</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>AAPL</td>
      <td>2.865508e+12</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>MSFT</td>
      <td>3.016308e+12</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>WPC</td>
      <td>3.098176e+12</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>RENT</td>
      <td>8.506228e+12</td>
    </tr>
  </tbody>
</table>
<p>3406422 rows × 2 columns</p>
</div>

```python
# To calculate the sector scores, we merge the sectors dataframe with the main dataframe
sector_scores = (
    df.reset_index()
    .merge(sectors, on="symbol")
    .set_index("date")[["symbol"] + sectors.columns.tolist()]
)
sector_scores
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>sector_basic_materials</th>
      <th>sector_business_services</th>
      <th>sector_consumer_cyclical</th>
      <th>sector_consumer_defensive</th>
      <th>sector_energy</th>
      <th>sector_financial_services</th>
      <th>sector_healthcare</th>
      <th>sector_industrials</th>
      <th>sector_other</th>
      <th>sector_real_estate</th>
      <th>sector_technology</th>
      <th>sector_utilities</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-01</th>
      <td>EOSS</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>YTEN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>VISL</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>ASLE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>ALBO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>NLY</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>AAPL</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>MSFT</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>WPC</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>RENT</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3393141 rows × 13 columns</p>
</div>

```python
# Size factor calculation
size_df = factor_sze(df, lower_decile=0.2, upper_decile=0.8)

# Momentum factor calculation
mom_df = factor_mom(df, trailing_days=252, winsor_factor=0.01)

# Value factor calculation
value_df = factor_val(df, winsor_factor=0.05)
```

```python
# Momentum distribution
mom_df["mom_score"].hist(bins=100)
plt.title("Momentum Factor Distribution")
plt.show()

# Value distribution
value_df["val_score"].hist(bins=100)
plt.title("Value Factor Distribution")
plt.show()
```

![demo_9_0](https://github.com/user-attachments/assets/37fbfa7c-1937-49b7-a557-c6c8d8b2cd25)
![demo_9_1](https://github.com/user-attachments/assets/4ae80b49-7f7b-428b-b67d-bd60f55aff91)

```python
# %%  All of which we merge to get the style scores
style_scores = (
    value_df.merge(mom_df, on=["symbol", "date"])
    .merge(size_df, on=["symbol", "date"])
    .dropna()
)
style_scores
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symbol</th>
      <th>val_score</th>
      <th>mom_score</th>
      <th>sze_score</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-28</th>
      <td>ALBO</td>
      <td>2.732799</td>
      <td>-0.308748</td>
      <td>3.418444</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>HSDT</td>
      <td>2.441402</td>
      <td>-1.777434</td>
      <td>3.316239</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>ASLE</td>
      <td>1.547939</td>
      <td>0.836804</td>
      <td>3.259800</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>VISL</td>
      <td>2.732799</td>
      <td>-1.571045</td>
      <td>3.042578</td>
    </tr>
    <tr>
      <th>2020-04-28</th>
      <td>KNWN</td>
      <td>1.570320</td>
      <td>-0.325302</td>
      <td>2.970678</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>NLY</td>
      <td>-2.076270</td>
      <td>0.015212</td>
      <td>-2.906904</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>AAPL</td>
      <td>-1.156765</td>
      <td>0.344759</td>
      <td>-2.996094</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>MSFT</td>
      <td>-0.963887</td>
      <td>0.876672</td>
      <td>-3.016549</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>WPC</td>
      <td>-2.877029</td>
      <td>-0.090848</td>
      <td>-3.027231</td>
    </tr>
    <tr>
      <th>2024-02-16</th>
      <td>RENT</td>
      <td>-2.327684</td>
      <td>-2.079591</td>
      <td>-3.430058</td>
    </tr>
  </tbody>
</table>
<p>2479098 rows × 4 columns</p>
</div>

```python
ddf = (
    ret_df.merge(cap_df, on=["date", "symbol"])
    .merge(sector_scores, on=["date", "symbol"])
    .merge(style_scores, on=["date", "symbol"])
    .dropna()
    .astype({"symbol": "category"})
)
returns_df = ddf[["symbol", "asset_returns"]]
mkt_cap_df = ddf[["symbol", "market_cap"]]
sector_df = ddf[["symbol"] + sectors.columns.tolist()]
style_df = ddf[style_scores.columns]
```

```python
fac_df, eps_df = estimate_factor_returns(
    returns_df,
    mkt_cap_df,
    sector_df,
    style_df,
    winsor_factor=0.1,
    residualize_styles=False,
)
```

```python
factor_cols = ["market","mom_score", "val_score", "sze_score"]

fac_df[factor_cols].plot(subplots=True, figsize=(15, 10))
plt.title("Factor Returns")
plt.show()

(fac_df[factor_cols] + 1).cumprod().plot(figsize=(15, 10))
plt.title("Factor Returns Cumulative")
plt.show()
```

![demo_13_1](https://github.com/user-attachments/assets/4f47325c-ce8b-42c3-867f-b4159f581f93)

![demo_13_0](https://github.com/user-attachments/assets/fe9cd708-7b7f-49dd-9c5f-362fa238b055)

```python
y = returns_df.query("symbol == 'AAPL'")["asset_returns"]
X = fac_df[["market", "sector_technology", "mom_score", "val_score", "sze_score"]]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit().get_robustcov_results()
results.summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>asset_returns</td>  <th>  R-squared:         </th> <td>   0.610</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.608</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   259.8</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 21 Feb 2025</td> <th>  Prob (F-statistic):</th> <td>3.77e-175</td>
</tr>
<tr>
  <th>Time:</th>                 <td>10:42:32</td>     <th>  Log-Likelihood:    </th> <td>  2914.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   959</td>      <th>  AIC:               </th> <td>  -5817.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   953</td>      <th>  BIC:               </th> <td>  -5787.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>         <td>HC1</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>             <td>    0.0004</td> <td>    0.000</td> <td>    0.955</td> <td> 0.340</td> <td>   -0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>market</th>            <td>   -1.3258</td> <td>    0.175</td> <td>   -7.594</td> <td> 0.000</td> <td>   -1.668</td> <td>   -0.983</td>
</tr>
<tr>
  <th>sector_technology</th> <td>    0.3238</td> <td>    0.087</td> <td>    3.738</td> <td> 0.000</td> <td>    0.154</td> <td>    0.494</td>
</tr>
<tr>
  <th>mom_score</th>         <td>    0.4925</td> <td>    0.095</td> <td>    5.211</td> <td> 0.000</td> <td>    0.307</td> <td>    0.678</td>
</tr>
<tr>
  <th>val_score</th>         <td>   -1.4278</td> <td>    0.233</td> <td>   -6.140</td> <td> 0.000</td> <td>   -1.884</td> <td>   -0.971</td>
</tr>
<tr>
  <th>sze_score</th>         <td>   -4.6804</td> <td>    0.389</td> <td>  -12.030</td> <td> 0.000</td> <td>   -5.444</td> <td>   -3.917</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>241.840</td> <th>  Durbin-Watson:     </th> <td>   1.897</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2493.520</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.841</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.718</td>  <th>  Cond. No.          </th> <td>1.14e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are heteroscedasticity robust (HC1)<br/>[2] The condition number is large, 1.14e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.

**The strong multicollinearity is due to the estimated market factor derived from the sectors is almost correlated 1:1 with the sze factor.**
