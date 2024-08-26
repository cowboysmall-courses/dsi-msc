

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 -
plt.figure(figsize = (8, 6))
plt.style.use("ggplot")

sns.set_style("darkgrid")
sns.set_context("paper")



# %% 1 - retrieve data for DJIA
indices    = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].to_list()
start_date = "2019-01-01"
end_date   = "2023-12-31"



# %% 2 - retrieve data
data = yf.download(indices, "2018-12-31", "2024-01-01", progress = False)



# %% 3 - collect column names with data
rows = data["Close"].shape[0]
cols = []

for column in data["Close"].columns:
    if data["Close"][column].isna().sum() / rows != 0:
        cols.append(column) 



# %% 4 - impute missing data using LOCF (forward fill)
data_adjc = data["Adj Close"].drop(columns = cols)
data_rets = pd.DataFrame({index: data_adjc[index].pct_change() * 100 for index in data_adjc.columns})



# %% 3 - impute missing data using LOCF (forward fill)
data_adjc = data_adjc.ffill()
data_adjc = data_adjc[start_date:end_date]



# %% 3 - impute missing data using LOCF (forward fill)
data_rets = data_rets.ffill()
data_rets = data_rets[start_date:end_date]



# %% 3 - 
def scaled(data):
    scaled = (data - data.min()) / (data.max() - data.min())
    return scaled.T 



# %% 3 - 
data_rets_scaled = scaled(data_rets)

