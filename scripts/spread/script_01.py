

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import yfinance as yf

from itertools import combinations

from statsmodels.tsa.stattools import coint

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from cowboysmall.data import snp500

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



