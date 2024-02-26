import pandas as pd
import os
from algo_mesoplodon_bowdoini import Trader

#Utils:
def MidPriceAndReturns(df: pd.DataFrame):
    df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2
    df["returns"] = df["mid_price"].pct_change()

#------------------------------------------------------------------
CWD = os.getcwd()
DATA = os.path.join(CWD,"data")

files = []
for i in os.listdir(DATA):
    f = os.path.join(DATA,i)
    files.append(f)

#tutorial data test------------------------------------------------
tutorial_data = pd.read_csv(files[0], sep=";")
dfs = {}
AMETHYSTS_DF = tutorial_data[tutorial_data["product"]=="AMETHYSTS"]
STARFRUIT_DF = tutorial_data[tutorial_data["product"]=="STARFRUIT"]
dfs["AMETHYSTS"] = AMETHYSTS_DF
dfs["STARFRUIT"] = STARFRUIT_DF


for i in dfs.keys():
    df = pd.DataFrame(dfs.get(i))
    MidPriceAndReturns(df=df) 

    '''NEXT: make a class backtester that takes all positions and computes returns'''