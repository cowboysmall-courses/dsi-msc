

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import yfinance as yf

from collections import Counter

from itertools import combinations

from statsmodels.tsa.stattools import coint

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from cowboysmall.data import snp500

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 - retrieve data for DJIA
indices = snp500.read()["Symbol"].to_list()



# %% 2 - retrieve data for indices
data = yf.download(indices, "2017-12-31", "2024-01-01", progress = False)



# %% 2 - 
adjc = data["Close"].dropna(axis = 1)
rets = adjc.pct_change().iloc[1:, :]



# %% 3 - 
rets["AAPL"].plot(figsize = (16, 6))
rets.head()



# %% 5 - 
pca = PCA(n_components = 50)
pca.fit(rets)



# %% 5 - 
cmp = pd.DataFrame(pca.components_, columns = rets.columns)
cmp.head()



# %% 5 - 
X = StandardScaler().fit_transform(cmp).T



# %% 5 - 
clst = KMeans(n_clusters = 25, max_iter = 1000)
clst.fit(X)



# %% 5 - 
lbls = clst.labels_
print(np.bincount(lbls))

cntr = Counter(lbls)
lbls_p = [cntr[v] <  100 for v in lbls]
lbls_n = [cntr[v] >= 100 for v in lbls]



# %% 5 - 
tsne = TSNE().fit_transform(X)



# %% 5 - 
plt.figure(figsize = (16, 9))
plt.title("Clusters: KMeans")

plt.scatter(tsne[(lbls_p), 0], tsne[(lbls_p), 1], alpha = 0.95, c = lbls[lbls_p], cmap = cm.Paired)
plt.scatter(tsne[(lbls_n), 0], tsne[(lbls_n), 1], alpha = 0.05)

plt.axis(False)
plt.show()



# %% 3 - 
def nCr(n, r):
    return n * (n - 1) // r



# %% 3 - 
for lbl in sorted(list(set(lbls))):
    if lbl > -1:
        rows = rets.T.loc[(lbls == lbl), :]

        cint = 0
        for i1, i2 in combinations(rows.index.values, 2):
            result = coint(rows.loc[i1], rows.loc[i2])
            if result[1] < 0.05:
                cint += 1

        print()
        print(f"          cluster: {lbl + 1}")
        print(f"             size: {rows.shape[0]}")
        print(f" all cointegrated: {cint == nCr(rows.shape[0], 2)}")
        print()



# %%
