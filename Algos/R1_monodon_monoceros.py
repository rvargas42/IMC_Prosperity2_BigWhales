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
		"STARFRUIT":{},
		"AMETHYSTS":{},
		"ORCHIDS":{},
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
		def monoceros(product, order_depth, i, depth, orders, maxSize, MaxDepth):
			print("inside quote model")
			symetry = []
			depth_symetry = 0
			buys, sells = order_depth.buy_orders, order_depth.sell_orders
			best_bid, best_bid_Q = list(buys.items())[0]
			best_ask, best_ask_Q = list(sells.items())[0]
			optBid, optAsk = 0, 0
			Qt = 0
			for i in range(depth):
				level_bid_Q = list(buys.items())[i][1]
				level_ask_Q = list(sells.items())[i][1]
				Qt = np.abs(level_ask_Q) + np.abs(level_bid_Q)
				sym = level_bid_Q + level_ask_Q
				symetry.append(sym)
			for i, _ in enumerate(symetry):
				depth_symetry += symetry[i] * (i+1)
			total_symetry = sum(symetry)
			print("symetry and depth sym: ", total_symetry, depth_symetry)
			reservation_level = np.min([np.abs(total_symetry - total_symetry), MaxDepth])
			Lb = reservation_level * np.abs(i) if i > 0 else reservation_level * (1+np.abs(i))
			La = reservation_level * (1+np.abs(i)) if i < 0 else reservation_level * np.abs(i)
			optBid, optAsk = best_bid - Lb, best_ask + La
			Qa = -np.abs(total_symetry-maxSize[0]) if total_symetry > 0 else maxSize[0]
			Qb = maxSize[1] if total_symetry > 0 else np.abs(total_symetry+maxSize[1])
			orders.append(L:=Order(product, int(optBid), int(Qb)))
			orders.append(S:=Order(product, int(optAsk), int(Qa)))
			print("mid_price: ", (best_ask+best_bid)/2)
			print("askquote bidquote: ", optAsk, optBid)
			print("Long/short orders; ", L, S)
			# if i == 0:
			# 	BidQ, AskQ = maxSize[1] * (1-i), maxSize[0] * i
			# 	orders.append(L:=Order(product, int(optBid), int(BidQ)))
			# 	orders.append(S:=Order(product, int(optAsk), int(AskQ)))
			# 	print("mid_price: ", (best_ask+best_bid)/2)
			# 	print("askquote bidquote: ", optAsk, optBid)
			# 	print("Long/short orders; ", L, S)
			# if i > 0:
			# 	BidQ, AskQ = maxSize[1] * (1-i), maxSize[0] * i
			# 	orders.append(L:=Order(product, int(optBid), int(BidQ)))
			# 	orders.append(S:=Order(product, int(optAsk), int(AskQ)))
			# 	print("mid_price: ", (best_ask+best_bid)/2)
			# 	print("askquote bidquote: ", optAsk, optBid)
			# 	print("Long/short orders; ", L, S)
			# if i < 0:
			# 	BidQ, AskQ = maxSize[1] * i, maxSize[0] * (1+i)
			# 	orders.append(L:=Order(product, int(optBid), int(BidQ)))
			# 	orders.append(S:=Order(product, int(optAsk), int(AskQ)))
			# 	print("mid_price: ", (best_ask+best_bid)/2)
			# 	print("askquote bidquote: ", optAsk, optBid)
			# 	print("Long/short orders; ", L, S)

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
	def tradeSTARFRUIT(state, result):
		'''
		'''
		STARFRUIT = "STARFRUIT"
		order_depth: OrderDepth = state.order_depths[STARFRUIT]
		depth : int = Trader.Utils.getDepth(order_depth)
		maxOrderSize = Trader.Utils.maxOrderSize(STARFRUIT, state)
		OBI = Trader.Utils.OrderBookImbalance(order_depth)
		print("OBI: ", OBI)
		print("maxOrderSize; ", maxOrderSize)

		orders : List[Order] = []
		Trader.Models.monoceros(
			STARFRUIT,
			order_depth,
			OBI,
			depth+1,
			orders,
			maxOrderSize,
			MaxDepth=3
		)

		result[STARFRUIT] = orders
		return result
	
	def run(self, state: TradingState):

		result : Dict = {}
		conversions = 1

		Trader.tradeSTARFRUIT(state, result)
		Trader.tradeAMETHYSTS(state, result)

		#Trader.DATA["t"] += 1
		traderData = "" #jp.encode(Trader.DATA)

		return result, conversions, traderData