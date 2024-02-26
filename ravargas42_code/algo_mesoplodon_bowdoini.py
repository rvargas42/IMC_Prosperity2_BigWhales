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

	_Long_limit: int = 10
	_Short_limit: int = -10

	class Utils:

		def _Spread(self, order_depth: OrderDepth) -> int:
			best_ask: int = (order_depth.sell_orders.items())[0]
			best_bid: int = (order_depth.buy_orders.items())[0]
			S: int = best_ask - best_bid

		def _MidPrice(self, order_depth: OrderDepth) -> int:
			best_ask: int = (order_depth.sell_orders.items())[0]
			best_bid: int = (order_depth.buy_orders.items())[0]
			MP : int = int((best_ask + best_bid) / 2)

			return MP
		
		def _OrderBookImbalance(self, order_depth: OrderDepth) -> Dict[str, int]:
			'''
			references: https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6
			'''
			numerator, denominator, OBI = 0, 0, 0
			# L -> max depth of market.
			L : int = np.min([len(order_depth.buy_orders), len(order_depth.buy_orders)]) #Dict[int,int] = {9:10, 10:11, 11:4}
			# Calculate imbalance:
			for i in range(L):
				buy_level_Q = list(order_depth.buy_orders.values())[i]
				sell_level_Q = list(order_depth.sell_orders.values())[i]
				numerator += buy_level_Q - sell_level_Q
				denominator += buy_level_Q + sell_level_Q
				if i == L:
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
			# Initialize the list of Orders to be sent as an empty list
			orders: List[Order] = []
			# Define a fair value for the PRODUCT. Might be different for each tradable item
			# Note that this value of 10 is just a dummy value, you should likely change it!
			acceptable_price = 10
						# All print statements output will be delivered inside test results
			print("Acceptable price : " + str(acceptable_price))
			print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
	
						# Order depth list come already sorted. 
						# We can simply pick first item to check first item to get best bid or offer
			if len(order_depth.sell_orders) != 0:
				best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
				if int(best_ask) < acceptable_price:
					# In case the lowest ask is lower than our fair value,
					# This presents an opportunity for us to buy cheaply
					# The code below therefore sends a BUY order at the price level of the ask,
					# with the same quantity
					# We expect this order to trade with the sell order
					print("BUY", str(-best_ask_amount) + "x", best_ask)
					orders.append(Order(product, best_ask, -best_ask_amount))
	
			if len(order_depth.buy_orders) != 0:
				best_bid, best_bid_amount = list(order_depth.buy_orders.items())[2]
				if int(best_bid_amount) >= 24:
										# Similar situation with sell orders
					print("SELL", str(best_bid_amount) + "x", best_bid)
					orders.append(Order(product, best_bid, -best_bid_amount))
			
			result[product] = orders
	
			# String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
		traderData = "SAMPLE" 
		
				# Sample conversion request. Check more details below. 
		conversions = 1
		return result, conversions, traderData