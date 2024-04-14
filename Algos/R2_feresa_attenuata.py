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
import math as mt
import numpy as np
import pandas as pd
import statistics as st
import jsonpickle as jp

class Trader:
	
	'''
	DATA: matrix containing historic data for all products and features
	shape(3,2,100) -> (3) dimensions or products
			  -> (2) features: mid_price, kalman predictions,
			  -> (100) entries: lenght of data 
	'''
	# ------------------------------- Encoded Data ------------------------------- #
	# 1 dimension per product (3), number of features: midprice,kf_pred,... / length
	# -------------------------------- Helper Data ------------------------------- #
	PRODUCTS = {"STARFRUIT":0,"AMETHYSTS":1,"ORCHIDS":2}
	LIMITS = {
		"STARFRUIT": 20,
		"AMETHYSTS": 20,
		"ORCHIDS": 100,
	}
	OPTIMUM_W = {
		"STARFRUIT": 0.261,
		"AMETHYSTS": 0.739
	}
	# ----------------- Utils class with tools to manipulate data ---------------- #
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
			best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
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
		def optimalInventory(data, position, product) -> tuple:
			current_position : int = np.abs(position.get(product, 0))
			desired_position : int = int(Trader.OPTIMUM_W[product] * 20)
			q = current_position - desired_position
			return q
		
		@staticmethod
		def positionsTracker(state, data): #TODO: create a list of filled trades by price so that they are closed when current price is above/bellow market
			'''constructs and order tracking data structure to be able to get in/out of trades correctly'''
			pass

		@staticmethod
		def savitzky_golay(y, window_size=11, order=3, deriv=0, rate=1): #TODO: implement a model that takes the gradiend of the predictions and then filters it to take leading indicator
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
			m = np.linalg.pinv(b).A[deriv] * rate**deriv * mt.factorial(deriv)
			# pad the signal at the extremes with
			# values taken from the signal itself
			firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
			lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
			y = np.concatenate((firstvals, y, lastvals))
			preds = np.convolve( m[::-1], y, mode='valid')
			y_hat = np.int32(np.round(preds,0))
			return y_hat

	# --------- Class contianing quoting, filtering and any kind of model -------- #
	class Models:
		@staticmethod
		def monoceros(product, order_depth, i, depth, orders, maxSize, MaxDepth):
			symetry = []
			depth_symetry = 0
			buys, sells = order_depth.buy_orders, order_depth.sell_orders
			best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
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
			reservation_level = np.min([np.abs(total_symetry - total_symetry), MaxDepth])
			Lb = reservation_level * np.abs(i) if i > 0 else reservation_level * (1+np.abs(i))
			La = reservation_level * (1+np.abs(i)) if i < 0 else reservation_level * np.abs(i)
			optBid, optAsk = best_bid - Lb, best_ask + La
			Qa = -np.abs(total_symetry-maxSize[0]) if total_symetry > 0 else maxSize[0]
			Qb = maxSize[1] if total_symetry > 0 else np.abs(total_symetry+maxSize[1])
			orders.append(L:=Order(product, int(optBid), int(Qb)))
			orders.append(S:=Order(product, int(optAsk), int(Qa)))

	def tradeAMETHYSTS(self):
		'''Amethysts seem to be capped between two prices: [9996.5 - 10003.5] which is a spread of 7$SH
		Strategy: Calculate a mean price and sell or buy given a distance to price and OBI
		'''
		orders : List[Order] = []
		AMETHYSTS = "AMETHYSTS"
		order_depth: OrderDepth = self.state.order_depths[AMETHYSTS]
		BookImbalance = self.Utils.OrderBookImbalance(order_depth)
		maxOrderSize = self.maxOrderSize(AMETHYSTS)

		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = self.Utils.midPrice(order_depth)

		reservation_bid, reservation_ask, mean = 9999, 10001, 10000
		
		if mid_price < reservation_bid:
			order = Order(AMETHYSTS, reservation_bid, int(maxOrderSize[1]))
			print("buy order at: ",  reservation_bid, maxOrderSize[1])
			orders.append(order)
		if mid_price > reservation_ask:
			order = Order(AMETHYSTS, reservation_ask, int(maxOrderSize[0]))
			print("sell order at: ",  reservation_bid, maxOrderSize[0])
			orders.append(order)

		self.result[AMETHYSTS].extend(orders)

	def tradeSTARFRUIT(self):
		'''
		'''
		STARFRUIT, data_key = "STARFRUIT", self.PRODUCTS["STARFRUIT"]
		order_depth: OrderDepth = self.state.order_depths[STARFRUIT]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		depth : int = self.Utils.getDepth(order_depth)
		maxOrderSize = self.maxOrderSize(STARFRUIT)
		mid_price = self.DATA[data_key][0][-1]
		fair_price = self.DATA[data_key][1][0]
		print(fair_price)
		print(mid_price)
		orders : List[Order] = []

		buyQ = np.random.randint(1,3)
		sellQ = np.random.randint(-3,-1)

		if (best_ask < fair_price and best_bid < fair_price):
			order = Order(STARFRUIT, best_ask, buyQ)
			orders.append(order)
		if (best_bid > fair_price and best_ask > fair_price):
			order = Order(STARFRUIT, best_bid, sellQ)

		self.result[STARFRUIT].extend(orders)

	def tradeORCHIDS(self):
		'''
		HUMIDIY: 60-80% is the ideal range->if it goes above or bellow-> Production decreases 2% for every 5% humidity change
		SUNLIGHT: Has to be >7hours/day -> if not then Production decreases -4%/10 minutes
		max position should be 100
		The idea for this product is to predict production levels or prices and then put orders above or bellow market
		'''
		ORCHIDS = "ORCHIDS"
		# ------------------------ Data from the South Island ------------------------ #
		humidity = self.state.observations.conversionObservations[ORCHIDS].humidity
		sunlight = self.state.observations.conversionObservations[ORCHIDS].sunlight
		costOfSale = self.state.observations.conversionObservations[ORCHIDS].exportTariff
		costOfBuy = self.state.observations.conversionObservations[ORCHIDS].importTariff
		fees = self.state.observations.conversionObservations[ORCHIDS].transportFees
		bid = self.state.observations.conversionObservations[ORCHIDS].bidPrice
		ask = self.state.observations.conversionObservations[ORCHIDS].askPrice
		# -------------------------- BigWhale Island Market -------------------------- #
		order_depth: OrderDepth = self.state.order_depths[ORCHIDS]


		orders : List[Order] = []

		self.result[ORCHIDS].extend(orders)

	def maxOrderSize(self, product):
		limit = Trader.LIMITS[product]
		productPosition = self.state.position.get(product,0)
		if np.abs(productPosition) > 20:
			return (0, 0)
		shortSize : int = int(-1 * (productPosition + limit))
		longSize : int = int(-1 * (productPosition - limit))
		return (shortSize, longSize)

	def updateData(self): #OK!
		'''
		Method to populate data matrix for keeping track of historic data during execution
		- Features: mid_price, kalman filter, 
		'''
		#fill values from Data array with mean where values are 0 to predict correctly (similar to ffill)
		if self.time == 2:
			mean = np.mean(self.DATA[self.DATA != 0])
			self.DATA[self.DATA == 0] = mean
		else:
			for i, prod in enumerate(self.PRODUCTS.items()):
				mid_price = Trader.Utils.midPrice(self.state.order_depths[prod[0]])
				time = int(np.min([self.time,99]))
				if time < 99:
					self.DATA[i][0][time] = mid_price
					if time > 11:
						preds = Trader.Utils.savitzky_golay(self.DATA[i][0])
						self.DATA[i][1] = preds
				if time == 99:
					self.DATA[i][0] = np.roll(self.DATA[i][0],-1)
					self.DATA[i][0][time] = mid_price
					preds = Trader.Utils.savitzky_golay(self.DATA[i][0]) #TODO - make predictions make after 11 timestamp
					self.DATA[i][1] = preds
					continue
				#after updating midprice we predict for the whole 100 entries

	def calculatePosition(self, takeProfit=0.01, stopLoss=0.009):
		'''
		Method that takes OPEN_POSITIONS first calculates profitability
		of each position and exteds results with close_orders
		close_orders are orders that zero out long/short positions at a
		given price level. i.e: long +5 @ 100 would be closed with short -5 @ market
		'''
		# ------------ 1. Enter positions and perform profit calculations ------------ #
		for product, orders in self.OPEN_POSITIONS.items():
			close_orders = []
			order_depth = self.state.order_depths[product]
			position = self.state.position.get(product, 0)
			best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
			returns = 0
			for price, quant in orders.items():
				market_price = best_ask if quant > 0 else best_bid
				profit = ((market_price - price)/price) if quant >= 0 else ((market_price - price)/market_price)
				returns += profit * quant
			# ---------------------- Closing Orders at market price: --------------------- #
			if takeProfit <= returns or returns <= -stopLoss:
				order = Order(product, market_price, -position)
				close_orders.append(order)

			self.result[product].extend(close_orders) #this will be the orders that zero out our position

	def updatePositions(self): # May be working...
		'''
		Updates self.OPEN_POSITIONS with new filled orders for each product
		'''
		own_trades = self.state.own_trades
		if not own_trades:
			return
		for product, trades in self.OPEN_POSITIONS.items():
			maxShort, maxLong = self.maxOrderSize(product)
			print("maxordersize: ",maxShort,maxLong)
			position = self.state.position.get(product,0)
			trade_list = own_trades.get(product, 0)
			if position == 0 or not trade_list:
				trades.clear() #means that all trades zero out each other
				continue
			for trade in trade_list:
				if int(trade.timestamp / 100) == self.time - 1:#Check for Trade objects that are pertinent to previous tstamp
					P = int(trade.price)
					Q = int(-trade.quantity) if trade.seller == "SUBMISSION" else int(trade.quantity)
					if Q < 0:
						maxQ = np.max([maxShort, Q])
						trades[P] = trades.get(P, 0)
						if trades[P] < 0:
							trades[P] = maxQ
						else:
							trades[P] += Q
					else:
						maxQ = np.min([maxLong, Q])
						trades[P] = trades.get(P, 0)
						if trades[P] > 0:
							trades[P] = maxQ
						else:
							trades[P] += Q
				else:
					continue
			#NOTE - This method nets out positions that are same quantity
			#self.deleteClosedOrders(trades)
			#NOTE - This next operations costs 0.02s but allows to faster operations if sorted
			trades = dict(sorted(trades.items()))
			self.OPEN_POSITIONS[product] = {key: value for key, value in trades.items() if value != 0}

	def run(self, state: TradingState):
		# ------------------------ Data Needed for operations ------------------------ #
		self.DATA = np.zeros((3,2,100))
		self.OPEN_POSITIONS = {prod:{} for prod in self.PRODUCTS.keys()}
		# -------------- Instance variables to be accessed and modified -------------- #
		self.result : Dict = {prod:[] for prod in self.PRODUCTS.keys()}
		self.state = state
		self.time = int(state.timestamp / 100)
		self.conversions = 1
		if not self.state.traderData:
			self.state.TraderData = self.DATA
		else:
			self.DATA, self.OPEN_POSITIONS = jp.decode(self.state.traderData, keys=True) #NOTE - OPEN_POSITIONS HAS KEYS AS STR AFTER DECODING
		self.updateData() #REVIEW - 
		print(self.DATA[0][0])
		print(self.DATA[0][1])
		# ------------------ 1. Calculate positions we need to close ----------------- #
		self.updatePositions() #REVIEW - 
		self.calculatePosition()
		# --------------------------- 2. Execute New Trades -------------------------- #

		#self.tradeAMETHYSTS()
		self.tradeSTARFRUIT()
		#self.tradeORCHIDS()

		# ------------------------ 3. Optimize Inventory Risk ------------------------ #
		#self.optimizeInventory()
		# ----------------------- 4. Enconde and Return Results ---------------------- #
		data2encode = (self.DATA, self.OPEN_POSITIONS)
		traderData = jp.encode(data2encode, keys=True)
		return self.result, self.conversions, traderData