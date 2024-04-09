import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from Utils.Config import CHARTS_DIR
import os

reservartion_price = 900
price_max = 1000
price_range = [i for i in range(reservartion_price, price_max)]

profit_curve = []
for Q, price_level in enumerate(price_range):

    COS = Q * price_level
    POS = 1000 * Q
    Real_Profit = Q/100 * (POS - COS)
    profit_curve.append(Real_Profit)

print(set(profit_curve))
max_profit = np.max(profit_curve)
new_min = 0
for i in range(66, 100):
    if profit_curve[i] < max_profit and profit_curve[i] > 1001:
        new_min = profit_curve[i]

min_profit = new_min
curve = plt.plot(profit_curve)
plt.xlabel("Lowest to Highest Bid")
plt.ylabel("Potential Profits")
plt.annotate(str(int(max_profit)), xy=(profit_curve.index(max_profit), max_profit))
plt.annotate(str(new_min), xy=(profit_curve.index(min_profit), min_profit))
plt.annotate(str(int(profit_curve.index(max_profit))+900), xy=(profit_curve.index(max_profit), 1))
plt.annotate(str(int(profit_curve.index(min_profit))+900), xy=(profit_curve.index(min_profit), 1))
plt.yticks(profit_curve, minor=True)
plt.axhline(y=max_profit)
plt.axvline(x=profit_curve.index(max_profit), color="green")
plt.axvline(x=profit_curve.index(min_profit), color="green")
title = "BUYING SCUBA GEAR"
plt.title(title)
plt.savefig(f"{CHARTS_DIR}/SCOOBAGEAR_MANUAL_Results_R1.jpg")