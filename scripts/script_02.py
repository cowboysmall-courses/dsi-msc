

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import yfinance as yf

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 -
plt.figure(figsize = (8, 6))
plt.style.use("ggplot")

sns.set_style("darkgrid")
sns.set_context("paper")



# %% 1 - retrieve data for DJIA
indices    = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].to_list()
start_date = "2019-1-1"
end_date   = "2023-12-31"



# %% 2 - retrieve data for indices
data = yf.download(indices, "2018-12-31", "2024-1-1", progress = False)



# %% 3 - collect column names with data
rows = data["Close"].shape[0]
cols = []

for column in data["Close"].columns:
    if data["Close"][column].isna().sum() / rows != 0:
        cols.append(column) 



# %% 3 - 
data_open = data["Open"].drop(columns = cols)
data_clse = data["Close"].drop(columns = cols)
data_rets = pd.DataFrame({index: data_clse[index].pct_change(fill_method = None) * 100 for index in data_clse.columns})



# %% 3 - impute missing data using LOCF (forward fill)
data_open = data_open.ffill()
data_open = data_open[start_date:end_date]



# %% 3 - impute missing data using LOCF (forward fill)
data_clse = data_clse.ffill()
data_clse = data_clse[start_date:end_date]



# %% 3 - impute missing data using LOCF (forward fill)
data_rets = data_rets.ffill()
data_rets = data_rets[start_date:end_date]



# %% 3 - 
def normed(data):
    normed = (data - data.mean()) / data.std()
    return normed.T 



# %% 3 - 
def cluster_plots(data, title):
    elbw = []
    sils = []

    K = list(range(2, 12))
    for i in  K:
        cluster = KMeans(n_clusters = i).fit(data)
        elbw.append(cluster.inertia_)
        sils.append(silhouette_score(data.values, cluster.labels_))

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))

    fig.suptitle(f"Cluster: {title}")

    axes[0].plot(K, elbw, 'bx-')
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("SSE")

    axes[1].plot(K, sils, 'bx-')
    axes[1].set_title("Silhouette Method")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Score")

    plt.show()




# %% 3 - 
data_rets_normed = normed(data_rets)
cluster_plots(data_rets_normed, "Returns - Normed")

data_rets_normed_cls = pd.DataFrame(KMeans(n_clusters = 4).fit_predict(data_rets_normed), columns = ["Cluster"], index = data_rets_normed.index)
print(np.bincount(data_rets_normed_cls["Cluster"]))



# %% 3 - 
data_clse_normed = normed(data_clse)
cluster_plots(data_clse_normed, "Close - Normed")

data_clse_normed_cls = pd.DataFrame(KMeans(n_clusters = 4).fit_predict(data_clse_normed), columns = ["Cluster"], index = data_clse_normed.index)
print(np.bincount(data_clse_normed_cls["Cluster"]))







# %% 4 - 



# %% 5 - 



# %% 6 - 



# %% 7 - 
