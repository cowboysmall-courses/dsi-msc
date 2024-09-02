

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cowboysmall.data import snp500

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 -
plt.style.use("ggplot")

sns.set_style("darkgrid")
sns.set_context("paper")



# %% 1 - retrieve data for DJIA
indices    = snp500.read()["Symbol"].to_list()
start_date = "2019-01-01"
end_date   = "2023-12-31"



# %% 2 - retrieve data for indices
data = yf.download(indices, "2018-12-31", "2024-01-01", progress = False)



# %% 3 - 
data_clse = data["Close"].dropna(axis = 1)
data_rets = pd.DataFrame({index: data_clse[index].pct_change() * 100 for index in data_clse.columns})



# %% 3 - 
def cluster_plots(data, title):
    elbw = []
    sils = []

    K = list(range(2, 12))
    for i in  K:
        cluster = KMeans(n_clusters = i).fit(data)
        elbw.append(cluster.inertia_)
        sils.append(silhouette_score(data, cluster.labels_))

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



# %% 3 - impute missing data using LOCF (forward fill)
data_clse = data_clse.ffill()
data_clse = data_clse[start_date:end_date]
data_clse = StandardScaler().fit_transform(data_clse).T

cluster_plots(data_clse, "Close - Normed")

print(np.bincount(KMeans(n_clusters = 4).fit_predict(data_clse)))



# %% 3 - impute missing data using LOCF (forward fill)
data_rets = data_rets.ffill()
data_rets = data_rets[start_date:end_date]
data_rets = StandardScaler().fit_transform(data_rets).T

cluster_plots(data_rets, "Returns - Normed")

print(np.bincount(KMeans(n_clusters = 4).fit_predict(data_rets)))



# %% 4 - 
