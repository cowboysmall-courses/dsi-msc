

# %% 0 - import required libraries
import pandas as pd
import numpy as np

import yfinance as yf

from itertools import combinations

from statsmodels.tsa.stattools import coint

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from cowboysmall.data import snp500

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 - retrieve data for DJIA
indices    = snp500.read()["Symbol"].to_list()
start_date = "2019-01-01"
end_date   = "2023-12-31"



# %% 2 - retrieve data
data = yf.download(indices, "2018-12-31", "2024-01-01", progress = False)



# %% 4 - impute missing data using LOCF (forward fill)
data_adjc = data["Adj Close"].dropna(axis = 1)
data_rets = pd.DataFrame({index: data_adjc[index].pct_change() * 100 for index in data_adjc.columns})



# %% 3 - impute missing data using LOCF (forward fill)
data_rets = data_rets.ffill()
data_rets = data_rets[start_date:end_date]
data_indx = data_rets.T.index
data_rets = MinMaxScaler().fit_transform(data_rets).T



# %% 3 - 
clusters = KMeans(n_clusters = 25).fit_predict(data_rets)
cluster  = data_rets[clusters == 14]
corr     = np.corrcoef(cluster)



# %% 3 - 
correlated  = []
for a1, a2 in combinations(range(cluster.shape[0]), 2):
    result = corr[a1, a2]
    if result > 0.7:
        correlated.append((round(result, 5), a1, a2))

print("\n".join(f"{t[0]}: {t[1]} - {t[2]}" for t in  sorted(correlated, reverse = True)))



# %% 3 - 
cointegrated = []
for a1, a2 in combinations(range(cluster.shape[0]), 2):
    result = coint(cluster[a1], cluster[a2])
    if result[1] < 0.05:
        cointegrated.append((result[1], a1, a2))

print("\n".join(f"{t[0]}: {t[1]} - {t[2]}" for t in  sorted(cointegrated)))



# %%
