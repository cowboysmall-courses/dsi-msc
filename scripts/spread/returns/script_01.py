

# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf


import warnings
warnings.filterwarnings("ignore", category = FutureWarning)



# %% 2 - retrieve data for indices
aapl = yf.download("AAPL", "2017-12-31", "2024-01-01", progress = False)
msft = yf.download("MSFT", "2017-12-31", "2024-01-01", progress = False)



# %% 2 - 
aapl["Returns"] = aapl["Close"].pct_change()
msft["Returns"] = msft["Close"].pct_change()

aapl = aapl[1:]
msft = msft[1:]



# %% 4 - 
X = msft["Returns"]
X = sm.add_constant(X)

Y = aapl["Returns"]



# %% 4 - 
result = sm.OLS(Y, X).fit()
print(result.summary())



# %% 4 - 
alpha = result.params.values[0]
beta  = result.params.values[1]

resi  = Y - (alpha + (beta * X["Returns"]))



# %% 4 - 
plt.figure(figsize = (16, 9))

plt.title("Spread of Returns")
plt.xlabel("Time")
plt.ylabel("Spread")

plt.plot(resi)

plt.show()



# %% 4 - 
