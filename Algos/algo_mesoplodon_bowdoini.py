"""
Author: ravargas.42t@gmail.com
algo_mesoplodon_bowdoini.py (c) 2024
Desc: description
Created:  2024-02-29T13:52:02.430Z
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

	class LinearRegression:
		pass
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
				K[k] = Pminus[k]/( Pminus[k]+R )
				xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
				P[k] = (1-K[k])*Pminus[k]

			return xhat

class MarketMakingModels:
	'''
	this models will compute and return the result variable in Trader.run()
	'''

	class AvellanedaStoikov:
		'''
		Classic model for basic ob modelling. It only takes inventory risk into account.
		:returns: Given an orderbook state it returns the set of bid ask orders.
		'''
		
		def _ReservationPrice(s,q,t, T, gamma, variance):
			#Calculate the reservation price based on target inventory.
			r = s - q * gamma * variance * (T - t)
			return r
		
		def _OptimalSpread(gamma, variance, T, t, k):
			#Determine the optimal bid and ask spread.

			delta = gamma * variance *(T - t) + (2/gamma) * np.log((1 + (gamma / k))) #where k is order book liquidity or depth (Levels)
			return delta

		def run(s, q, t, T, gamma, variance, k):
			print(f"{np.log(10)}")
			r, delta = (
				MarketMakingModels.AvellanedaStoikov._ReservationPrice(s, q, t, T, gamma, variance), 
				MarketMakingModels.AvellanedaStoikov._OptimalSpread(gamma, variance, T, t, k)
			)

			optimal_bid, optimal_ask = r - delta / 2, r + delta / 2

			return optimal_bid, optimal_ask
		
		def test():
			print("this is a test")

class Trader:

	_Long_limit: int = 0
	_Short_limit: int = 0


	def run(self, state: TradingState):
		"""
		main entry poiny of the script. It executes the rest of 
		methods to compute result, conversions and traderData
		"""
		print("traderData: " + state.traderData)
		print("Observations: " + str(state.observations))
		
				# Orders to be placed on exchange matching engine
		result = {}
		#for product in state.order_depths:
		product = "STARFRUIT"
		for product in state.order_depths:
			order_depth: OrderDepth = state.order_depths[product]
			Spread, MidPrice, OBI = Utils.TradeUtils.Spread(order_depth=order_depth), Utils.TradeUtils.MidPrice(order_depth=order_depth), Utils.TradeUtils.OrderBookImbalance(order_depth=order_depth)
			BuyDepth = len(order_depth.buy_orders)
			SellDepth = len(order_depth.sell_orders)
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
			
			# Initialize the list of Orders to be sent as an empty list
			orders: List[Order] = []
			# Define a fair value for the PRODUCT. Might be different for each tradable item
			acceptable_price = MidPrice * (1 + OBI)
			print("OBI: ", OBI)	
			print("Acceptable price : " + str(acceptable_price))
			print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

			if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
			
				if  OBI < 0.80:
					#for i in range(0,BuyDepth):
						BuyQ = -int(list(order_depth.buy_orders.values())[BuyDepth - 1])
						BuyP = list(order_depth.buy_orders.keys())[BuyDepth - 1]
						print("Buy", str(BuyP) + "x", BuyQ)
						order = Order(product, best_ask, best_ask_amount)
						orders.append(order)
						
				
				if OBI > 0.10:
					#for i in range(0,SellDepth):
						SellQ = -int(list(order_depth.sell_orders.values())[SellDepth - 1])
						SellP = list(order_depth.sell_orders.keys())[SellDepth - 1]
						print("BUY", str(SellQ) + "x", SellP)
						order = Order(product, best_bid, best_bid_amount)
						orders.append(order)

			traderData = "trading starfruit"

			result[product] = orders
	
			# String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
		
				# Sample conversion request. Check more details below. 
		conversions = 1
		return result, conversions, traderData