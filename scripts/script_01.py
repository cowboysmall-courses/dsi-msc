

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import yfinance as yf

from sklearn.preprocessing import scale
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
data.head()



# %% 3 - collect column names with data
rows = data["Close"].shape[0]
cols = []

for column in data["Close"].columns:
    if data["Close"][column].isna().sum() / rows == 0:
        cols.append(column) 



# %% 3 - 
data_open = data["Open"][cols]
data_clse = data["Close"][cols]
data_rets = pd.DataFrame({index: data_clse[index].pct_change(fill_method = None) * 100 for index in data_clse.columns})



# %% 3 - impute missing data using LOCF (forward fill)
data_open = data_open.ffill()
data_open = data_open[start_date:end_date]
data_open.head()



# %% 3 - impute missing data using LOCF (forward fill)
data_clse = data_clse.ffill()
data_clse = data_clse[start_date:end_date]
data_clse.head()



# %% 3 - impute missing data using LOCF (forward fill)
data_rets = data_rets.ffill()
data_rets = data_rets[start_date:end_date]
data_rets.head()



# %% 3 - 
# data_rets_norm = (data_rets - data_rets.mean()) / data_rets.std()
# data_rets_norm.head()

data_rets_norm = (data_rets - data_rets.min()) / (data_rets.max() - data_rets.min())
data_rets_norm = data_rets_norm.T
data_rets_norm.head()



# %% 3 - 
elbw = []
sils = []

K = list(range(2, 12))
for i in  K:
    cluster = KMeans(n_clusters = i).fit(data_rets_norm)
    elbw.append(cluster.inertia_)
    sils.append(silhouette_score(data_rets_norm.values, cluster.labels_))



# %% 3 - 
fig, axes = plt.subplots(1, 2, figsize = (16, 6))

fig.suptitle("Cluster: Returns")

axes[0].plot(K, elbw, 'bx-')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("SSE")

axes[1].plot(K, sils, 'bx-')
axes[1].set_title("Silhouette Method")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")

plt.show()



# %% 3 - 
data_clst = pd.DataFrame(KMeans(n_clusters = 5).fit_predict(data_rets_norm), columns = ["Cluster"], index = data_rets_norm.index)
data_clst.head()



# %% 3 - 
sizes = np.bincount(data_clst["Cluster"])
print(sizes)








# %% 3 - 
data_clse_norm = (data_clse - data_clse.min()) / (data_clse.max() - data_clse.min())
data_clse_norm = data_clse_norm.T
data_clse_norm.head()



# %% 3 - 
elbw = []
sils = []

K = list(range(2, 12))
for i in  K:
    cluster = KMeans(n_clusters = i).fit(data_clse_norm)
    elbw.append(cluster.inertia_)
    sils.append(silhouette_score(data_clse_norm.values, cluster.labels_))



# %% 3 - 
fig, axes = plt.subplots(1, 2, figsize = (16, 6))

fig.suptitle("Cluster: Close")

axes[0].plot(K, elbw, 'bx-')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("K")
axes[0].set_ylabel("SSE")

axes[1].plot(K, sils, 'bx-')
axes[1].set_title("Silhouette Method")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette Score")

plt.show()



# %% 3 - 
data_clst = pd.DataFrame(KMeans(n_clusters = 5).fit_predict(data_rets_norm), columns = ["Cluster"], index = data_rets_norm.index)
data_clst.head()



# %% 3 - 
sizes = np.bincount(data_clst["Cluster"])
print(sizes)




# %% 4 - merge data with outer join



# %% 5 - 



# %% 6 - 



# %%
