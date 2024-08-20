

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import yfinance as yf



# %% 1 - retrieve data for DJIA
INDICES    = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"].to_list()
START_DATA = '2019-1-1'
END_DATA   = '2023-12-31'


# %% 2 - retrieve data for indices
data = yf.download(INDICES, START_DATA, END_DATA, progress = False)
data.head()



# %% 3 - 
adjc = data["Adj Close"]
adjc.head()



# %% 4 - merge data with outer join
rows = adjc.shape[0]
dropped = []
for column in adjc.columns:
    pc = adjc[column].isna().sum() / rows
    if pc > 0:
        dropped.append(column) 



# %% 5 - 
adjc = adjc.drop(dropped, axis = 1)
adjc.head()


# %% 6 - impute missing data using LOCF (forward fill)
adjc = adjc.ffill()
adjc.head()




# %%
