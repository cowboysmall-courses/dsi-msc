
import pandas as pd

NIFTY_50_URL  = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
NIFTY_50_FILE = "./data/nifty50.csv"


def retrieve():
    pd.read_csv(NIFTY_50_URL).to_csv(NIFTY_50_FILE, index = False)


def read():
    return pd.read_csv(NIFTY_50_FILE)

