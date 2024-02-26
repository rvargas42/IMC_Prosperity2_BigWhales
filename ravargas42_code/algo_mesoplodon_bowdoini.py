#REQUIRED CLASSES
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
#Utils libraries
import math
import numpy as np
import pandas as pd
import statistics as st

class Trader:

	_Long_limit: int = 0
	_Short_limit: int = 0

	class Utils:

		@staticmethod
		def _InsertBookImbalance(df: pd.DataFrame) -> pd.DataFrame:
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
		def _Spread(order_depth: OrderDepth) -> int:
			best_ask: int = list(order_depth.sell_orders.keys())[0]
			best_bid: int = list(order_depth.buy_orders.keys())[0]
			S: int = best_ask - best_bid

			return S

		@staticmethod
		def _MidPrice(order_depth: OrderDepth) -> int:
			best_ask: int = list(order_depth.sell_orders.keys())[0]
			best_bid: int = list(order_depth.buy_orders.keys())[0]
			MP : int = int((best_ask + best_bid) / 2)

			return MP
		
		@staticmethod
		def _OrderBookImbalance(order_depth: OrderDepth) -> float:
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
				numerator += buy_level_Q - sell_level_Q
				denominator += buy_level_Q + sell_level_Q
				if i + 1 == L:
					OBI = numerator / denominator

			return OBI
	
	def run(self, state: TradingState):
		"""
		Only method required. It takes all buy and sell orders for all symbols as an input,
		and outputs a list of orders to be sent
		"""
		print("traderData: " + state.traderData)
		print("Observations: " + str(state.observations))
		
				# Orders to be placed on exchange matching engine
		result = {}
		for product in state.order_depths:
			order_depth: OrderDepth = state.order_depths[product]
			Spread, MidPrice, OBI = self.Utils._Spread(order_depth=order_depth), self.Utils._MidPrice(order_depth=order_depth), self.Utils._OrderBookImbalance(order_depth=order_depth)
			# Initialize the list of Orders to be sent as an empty list
			orders: List[Order] = []
			# Define a fair value for the PRODUCT. Might be different for each tradable item
			acceptable_price = MidPrice * (1 + OBI)
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

			print("OBI: ", OBI)			
			print("Acceptable price : " + str(acceptable_price))
			print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
	
			if len(order_depth.sell_orders) != 0:
				if OBI > 0.50:
					BuyQ = -int(best_bid_amount*np.abs(OBI))
					print("BUY", str(BuyQ) + "x", best_ask)
					orders.append(Order(product, best_bid - 1, BuyQ))
	
			if len(order_depth.buy_orders) != 0:
				if OBI < -0.50:
					SellQ = -int(best_ask_amount*np.abs(OBI))
					print("SELL", str(best_ask_amount) + "x", best_bid)
					orders.append(Order(product, best_ask + 1, SellQ))
			
			result[product] = orders
	
			# String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
		traderData = "Test"
		
				# Sample conversion request. Check more details below. 
		conversions = 1
		return result, conversions, traderData