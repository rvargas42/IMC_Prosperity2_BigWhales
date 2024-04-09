"""
Author: ravargas.42t@gmail.com
R1_monodon_monoceros.py (c) 2024
Desc:
	Round 1 algorithm to trade AMETHYSTS and STARFRUIT.
Created:  2024-04-09T09:02:23.136Z
Modified: !date!
"""

#REQUIRED CLASSES
from  datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
#support libraries
import math
import numpy as np
import pandas as pd
import statistics as st
import jsonpickle as jp

class Utils:

	'''
	This class contains all subclasses that will be used to perform
	operations on data inside the trader class
	'''

	class OBsimulator:

		@staticmethod
		def OrderImpact(order_depth: OrderDepth, order: Order) -> OrderDepth:
			'''
			This Method returns the simulated order book after a given order. It can
			be used to see the impact of an order; the idea is to minimize midprice shifts.
			'''
			new_book = OrderDepth()
			current_order_book = {**order_depth.buy_orders, **order_depth.sell_orders}
			orderQ = order.quantity
			orderP = order.price

			for n, item in enumerate(current_order_book.items()): #iterate to match price

				
				level_p, level_q = item

				if orderP == level_p:
								
					new_book[level_p] = level_q + orderQ #CHECK OTHER APPROACHES LIKE FIFO

				else:
					new_book[level_p] = level_q

			return new_book

	class TradeUtils:

		@staticmethod
		def InsertBookImbalance(df: pd.DataFrame) -> pd.DataFrame:
			'''
			tool to compute OBI withing dataframe
			'''
			def ComputeImbalance(buys, sells, L):
				numerator, denominator, OBI = 0, 0, 0
				for i in range(L):
					bid_Q = buys[f"bid_volume_{i+1}"]
					ask_Q = sells[f"ask_volume_{i+1}"]
					numerator += bid_Q - ask_Q
					denominator += bid_Q + ask_Q
					if i+1 == L:
						OBI = numerator / denominator
					return OBI
			buy_cols, ask_cols = [i for i in df.columns if "bid_volume" in i], [i for i in df.columns if "ask_volume" in i]
			OBI = []
			for i in range(0,len(df)):
				buys = df[buy_cols].iloc[i]
				buys_L = buys.notna().sum()
				sells = df[ask_cols].iloc[i]
				sells_L = sells.notna().sum()
				L = np.min([buys_L, sells_L])
				OBI.append(ComputeImbalance(buys, sells, L))
			df["order_book_imbalance"] = OBI

			return df

		@staticmethod
		def Spread(order_depth: OrderDepth) -> int:
			best_ask: int = list(order_depth.sell_orders.keys())[0]
			best_bid: int = list(order_depth.buy_orders.keys())[0]
			S: int = best_ask - best_bid

			return S

		@staticmethod
		def MidPrice(order_depth: OrderDepth) -> int:
			best_ask: int = list(order_depth.sell_orders.keys())[0]
			best_bid: int = list(order_depth.buy_orders.keys())[0]
			MP : int = int((best_ask + best_bid) / 2)

			return MP
		
		@staticmethod
		def OrderBookImbalance(order_depth: OrderDepth) -> float:
			'''
			references: https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6
			'''
			numerator, denominator, OBI = 0, 0, 0
			# L -> max depth of market.
			L : int = np.min([len(order_depth.buy_orders), len(order_depth.sell_orders)]) #Dict[int,int] = {9:10, 10:11, 11:4}
			# Calculate imbalance:
			for i in range(L):
				buy_level_Q = list(order_depth.buy_orders.values())[i]
				sell_level_Q = -1 * (list(order_depth.sell_orders.values())[i])
				numerator += buy_level_Q
				denominator += buy_level_Q + sell_level_Q
				if i + 1 == L:
					OBI = numerator / denominator

			return OBI

	class KalmanFilter:
		@staticmethod
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

class Trader:

	LIMITS = {
		"SF": 20,
		"AM": 20,
	}
	def tradeAMETHYSTS(self, state, result):
		AMETHYSTS = "AMETHYSTS"
		orders : List[Order] = []
		order_depth: OrderDepth = state.order_depths[AMETHYSTS]
		midprice = (order_depth.buy_orders.keys()[0] + order_depth.sell_orders.keys()[0]) / 2

		return result

	def tradeSTARFRUIT(self, state, result):
		STARFRUIT = "STARFRUIT"
		orders : List[Order] = []
		order_depth: OrderDepth = state.order_depths[STARFRUIT]
		midprice = (order_depth.buy_orders.keys()[0] + order_depth.sell_orders.keys()[0]) / 2

		return result
	
	def run(self, state: TradingState):

		result : Dict = {}
		conversions = 1
		products = list(state.listings.keys())
		decoded_traderData = jp.decode(state.traderData)

		if not decoded_traderData:
			state.traderData = {}
		self.tradeAMETHYSTS(state, result)
		self.tradeSTARFRUIT(state, result)

		traderData = jp.encode(state.traderData)

		return result, conversions, traderData