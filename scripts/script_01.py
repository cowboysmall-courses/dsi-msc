

# %% 1 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import yfinance as yf




# %% 2 - retrieve data for indices
def retrieve_data(index, start_date = '2017-1-1', end_date = '2023-12-31', progress = False):
    return yf.download(f'^{index}', start_date, end_date, progress = progress)



# %% 3 - 
INDICES = []

data    = [retrieve_data(index) for index in INDICES]



# %% 4 - merge data with outer join
merged = pd.concat(data, axis = 1)



# %% 5 - impute missing data using LOCF (forward fill)
merged.ffill(inplace = True)

