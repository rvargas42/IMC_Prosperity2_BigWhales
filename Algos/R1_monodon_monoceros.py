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

class Trader:

	LIMITS = {
		"STARFRUIT": 20,
		"AMETHYSTS": 20,
	}
	OPTIMUM_W = {
		"STARFRUIT": 0.261,
		"AMETHYSTS": 0.739
	}
	DATA = {
		"t":0,
		"STARFRUIT":[],
		"AMETHYSTS":[],
	}

	class Utils:
		@staticmethod
		def getDepth(order_depth: OrderDepth):
			buy_depth = len(order_depth.buy_orders.items())
			sell_depth = len(order_depth.sell_orders.items())
			depths = np.array((buy_depth, sell_depth))
			k = np.min(depths) - 1
			return k
		
		@staticmethod
		def midPrice(order_depth):
			best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
			best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
			mid = (best_ask + best_bid) / 2
			return mid

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
				sell_level_Q = list(order_depth.sell_orders.values())[i]
				numerator += buy_level_Q + sell_level_Q
				denominator += buy_level_Q - sell_level_Q
				if i + 1 == L:
					OBI = numerator / denominator

			return OBI

		@staticmethod
		def appendPrices(product, mid_price):
			p = Trader.DATA[product]
			p.append(mid_price)
		
		@staticmethod
		def maxOrderSize(product, state: TradingState):
			limit = Trader.LIMITS[product]
			productPosition = state.position.get(product,0)
			if np.abs(productPosition) > 20:
				return (0, 0)
			shortSize : int = int(-1 * (productPosition + limit))
			longSize : int = int(-1 * (productPosition - limit))
			return (shortSize, longSize)
	
		@staticmethod
		def optimalInventory(myData, position, product) -> tuple:
			current_position : int = np.abs(position.get(product, 0))
			desired_position : int = int(Trader.OPTIMUM_W[product] * 20)
			q = current_position - desired_position
			return q
			
	class Models:
		@staticmethod
		def monodon_monoceron(state, BookImbalance):
			pass
	
	@staticmethod
	def tradeAMETHYSTS(state, result):
		'''Amethysts seem to be capped between two prices: [9996.5 - 10003.5] which is a spread of 7$SH
		Strategy: Calculate a mean price and sell or buy given a distance to price and OBI
		'''
		orders : List[Order] = []
		AMETHYSTS = "AMETHYSTS"
		#inventory_distance = Trader.Utils.optimalInventory(myData, state.position)[AMETHYSTS]
		order_depth: OrderDepth = state.order_depths[AMETHYSTS]
		BookImbalance = Trader.Utils.OrderBookImbalance(order_depth)
		print("OBI: ", BookImbalance)
		maxOrderSize = Trader.Utils.maxOrderSize("AMETHYSTS", state)
		print("max_order_size: ", maxOrderSize)
		#Trader.Utils.appendMidPrice(AMETHYSTS, myData, midprice)
		#get best bid/ask
		best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
		best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
		mid_price = Trader.Utils.midPrice(order_depth)

		reservation_bid, reservation_ask, mean = 9999, 10001, 10000
		
		if mid_price < reservation_bid:
			order = Order(AMETHYSTS, reservation_bid, int(maxOrderSize[1]))
			orders.append(order)
		if mid_price > reservation_ask:
			order = Order(AMETHYSTS, reservation_ask, int(maxOrderSize[0]))
			orders.append(order)

		result[AMETHYSTS] = orders
		return result
	
	@staticmethod
	def tradeSTARFRUIT(state, result, myData):
		'''
		Strategy: use market making model
		'''
		STARFRUIT = "STARFRUIT"
		orders : List[Order] = []
		order_depth: OrderDepth = state.order_depths[STARFRUIT]
		depth : int = Trader.Utils.getDepth(order_depth)
		midprice = Trader.Utils.midPrice(order_depth)
		Trader.Utils.appendPrices(STARFRUIT, midprice)
		maxOrderSize = Trader.Utils.maxOrderSize(STARFRUIT, state)
		
		X_ask, X_bid = list(order_depth.sell_orders.items())[depth][0], list(order_depth.buy_orders.items())[depth][0]
		BookImbalance = Trader.Utils.OrderBookImbalance(order_depth)
		y_price = int(10 + X_bid * 0.5174 - 3.3103 * BookImbalance + 0.4806 * X_ask)
		print("STARFRUIT predicted Price: ", y_price)
		print("mid_price: ", midprice)

		best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
		best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
		volatility = np.std(myData[STARFRUIT])
		y = myData[STARFRUIT]
		X = [i for i in range(len(y))]
		slope = np.polyfit(x=X, y=y, deg=1)
		m = slope[0]

		if y_price > midprice:
			order = Order(STARFRUIT, int(best_bid - 2), int(maxOrderSize[1]))
			orders.append(Order)
		if y_price < midprice:
			order = Order(STARFRUIT, int(best_ask + 2), int(maxOrderSize[0]))
			orders.append(Order)

		result[STARFRUIT] = orders
		return result
	
	def run(self, state: TradingState):

		result : Dict = {}
		conversions = 1
		products = list(state.listings.keys())
		if state.traderData:
			myData = jp.decode(state.traderData)
		else:
			myData = Trader.DATA

		Trader.tradeSTARFRUIT(state, result, myData)
		Trader.tradeAMETHYSTS(state, result)
		
		Trader.DATA["t"] += 1
		traderData = jp.encode(Trader.DATA)

		return result, conversions, traderData