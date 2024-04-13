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

def KalmanFilter(data:np.array) -> np.array:
			# intial parameters
			n_iter = len(data)
			sz = (n_iter,) # size of array
			x = data.mean() # truth value or mean
			z = data # observations have to be normal

			Q = 1e-5 # process variance

			# allocate space for arrays
			xhat=np.zeros(sz)      # a posteri estimate of x
			P=np.zeros(sz)         # a posteri error estimate
			xhatminus=np.zeros(sz) # a priori estimate of x
			Pminus=np.zeros(sz)    # a priori error estimate
			K=np.zeros(sz)         # gain or blending factor

			variance = np.var(z)
				#optimal value
			R = 0.05**2 # estimate of measurement variance, change to see effect
			# intial guesses
			xhat[0] = 0.0
			P[0] = 1.0
			for k in range(1,n_iter):
				# time update
				xhatminus[k] = xhat[k-1]
				Pminus[k] = P[k-1]+Q
				# measurement update
				K[k] = Pminus[k] / (Pminus[k]+R)
				xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
				P[k] = (1-K[k])*Pminus[k]
			return xhat

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int16(window_size))
        order = np.abs(np.int16(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::1], y, mode='valid')