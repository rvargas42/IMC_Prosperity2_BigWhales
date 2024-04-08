"""
Author: ravargas.42t@gmail.com
dataUtils.py (c) 2024
Desc: help functions to arange data
Created:  2024-04-08T16:25:37.935Z
Modified: !date!
"""

import os
import pandas as pd
from Config import ROOT_DIR

def getDataPaths(round: int):
    data_files = os.listdir(os.path.join(ROOT_DIR, "Data", f"round_{round}"))
    data_files = [os.path.join(ROOT_DIR, "Data", f"round_{round}", i) for i in data_files if not ".zip" in i]
    return data_files

def concat_dfs(name, paths):
    dfs = []
    for f in paths:
        if name in f:
            dfs.append(pd.read_csv(f, sep=";"))
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    return concat_df


