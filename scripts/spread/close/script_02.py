

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
aapl["Relative Close"] = aapl["Close"] / aapl["Close"][0]
msft["Relative Close"] = msft["Close"] / msft["Close"][0]



# %% 4 - 
X = msft["Relative Close"]
X = sm.add_constant(X)

Y = aapl["Relative Close"]



# %% 4 - 
result = sm.OLS(Y, X).fit()
print(result.summary())



# %% 4 - 
alpha = result.params.values[0]
beta  = result.params.values[1]

resi  = Y - (alpha + (beta * X["Relative Close"]))



# %% 4 - 
plt.figure(figsize = (16, 9))

plt.title("Spread")
plt.xlabel("Time")
plt.ylabel("Spread")

plt.plot(resi)

plt.show()



# %% 3 - 
plt.figure(figsize = (16, 9))

plt.title("Close")
plt.xlabel("Time")
plt.ylabel("Close")

plt.plot(aapl["Close"], label = "AAPL")
plt.plot(msft["Close"], label = "MSFT")

plt.legend()
plt.show()



# %% 4 - 
plt.figure(figsize = (16, 9))

plt.title("Relative Close")
plt.xlabel("Time")
plt.ylabel("Close")

plt.plot(aapl["Relative Close"], label = "AAPL")
plt.plot(msft["Relative Close"], label = "MSFT")

plt.legend()
plt.show()



# %% 4 - 
