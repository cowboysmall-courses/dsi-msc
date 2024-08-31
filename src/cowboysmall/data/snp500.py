
import pandas as pd

SNP_500_URL  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SNP_500_FILE = "./data/snp500.csv"


def retrieve():
    pd.read_html(SNP_500_URL)[0].to_csv(SNP_500_FILE, index = False)


def read():
    return pd.read_csv(SNP_500_FILE)


