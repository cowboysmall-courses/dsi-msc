

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import statsmodels.api as sm

import yfinance as yf

from itertools import combinations

from statsmodels.tsa.stattools import coint

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 1 - retrieve data for DJIA
indices    = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].to_list()
start_date = "2018-01-01"
end_date   = "2023-12-31"



# %% 2 - retrieve data for indices
data = yf.download(indices, "2017-12-31", "2024-01-01", progress = False)



# %% 2 - 
adjc = data["Close"].dropna(axis = 1)
rets = adjc.pct_change().iloc[1:, :]



# %% 3 - 
rets["AAPL"].plot(figsize = (16, 6))



# %% 5 - 
pca = PCA(n_components = 50)
pca.fit(rets)



# %% 5 - 
X = preprocessing.StandardScaler().fit_transform(pca.components_.T)


# %% 5 - 
clst = DBSCAN(eps = 3)
clst.fit(X)


# %% 5 - 
lbls = clst.labels_
print(np.bincount(lbls + 1))




# %% 5 - 
tsne = TSNE().fit_transform(X)



# %% 5 - 
plt.figure(figsize = (16, 9))
plt.title("Clusters")

plt.scatter(tsne[(lbls != -1), 0], tsne[(lbls != -1), 1], s = 100, alpha = 0.95, c = lbls[lbls != -1], cmap = cm.Paired)
plt.scatter(tsne[(lbls == -1), 0], tsne[(lbls == -1), 1], s = 100, alpha = 0.05)

plt.axis("off")
plt.show()

