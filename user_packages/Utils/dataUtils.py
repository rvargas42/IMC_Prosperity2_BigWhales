"""
Author: ravargas.42t@gmail.com
dataUtils.py (c) 2024
Desc: help functions to arange data
Created:  2024-04-08T16:25:37.935Z
Modified: !date!
"""

import os
import pandas as pd
import numpy as np
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

def fft_filtering(df, feature):
    signal = df[feature].to_numpy()
    n = len(signal)
    dt = 1
    fhat = np.fft.rfft(signal, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(1, np.floor(n/2), dtype="int")
    #inverse transform
    indices = PSD > PSD[int(len(PSD)*0.05)]
    PSDclean = PSD * indices
    fhat = indices * fhat
    ffiltered = np.fft.ifft(fhat)

    return ffiltered

def euclideanDistance(self, s1: np.array, s2:np.array):
	subtract = s1 - s2
	distance = np.sqrt(np.dot(subtract.T, subtract))
	return distance