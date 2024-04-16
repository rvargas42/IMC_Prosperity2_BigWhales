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
	PRODUCTS = {"STARFRUIT":0,"AMETHYSTS":1,"ORCHIDS":2, "GIFT_BASKET": 3, "CHOCOLATE":4, "STRAWBERRIES": 5, "ROSES": 6}
	LIMITS = {
		"STARFRUIT": 20,
		"AMETHYSTS": 20,
		"ORCHIDS": 100,
		"CHOCOLATE":250,
		"STRAWBERRIES":350,
		"ROSES":60,
		"GIFT_BASKET":60,
	}
	OPTIMUM_W = {
		"STARFRUIT": 0.261,
		"AMETHYSTS": 0.739,
		"ORCHIDS":1,
		"BENCH_GIFT_BASKET":{
			"CHOCOLATE":4/11,
			"STRAWBERRIES":6/11,
			"ROSES":1/11
		},
		"BASKET": {
			"CHOCOLATE":0.33,
			"STRAWBERRIES":.33,
			"ROSES":0.34
		}
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
		def savitzky_golay(y, window_size=3, order=1, deriv=0, rate=1): #TODO: implement a model that takes the gradiend of the predictions and then filters it to take leading indicator
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
			return preds

	# --------- Class contianing quoting, filtering and any kind of model -------- #

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
			orders.append(order)
		if mid_price > reservation_ask:
			order = Order(AMETHYSTS, reservation_ask, int(maxOrderSize[0]))
			orders.append(order)

		self.result[AMETHYSTS].extend(orders)

	def tradeSTARFRUIT(self):
		'''

		'''
		STARFRUIT, data_key = "STARFRUIT", self.PRODUCTS["STARFRUIT"]
		order_depth: OrderDepth = self.state.order_depths[STARFRUIT]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		spread = best_ask - best_bid
		depth : int = self.Utils.getDepth(order_depth)
		data = self.DATA[data_key]
		maxShort, maxLong = self.maxOrderSize(STARFRUIT)
		mid_price = self.DATA[data_key][0][self.dataTime]
		print("mid_price: ",mid_price)
		# ---------------- Trade volatility with savitsky golay model ---------------- #
		fair_price = self.DATA[data_key][1][self.dataTime]
		print("mid_price: ",mid_price)

		orders = []
		if self.time > 0:
			trend = (self.DATA[data_key][0][self.dataTime] - self.DATA[data_key][0][0]) / self.DATA[data_key][0][0]
		else:
			return
		if best_ask < fair_price:
			orders.append(Order(STARFRUIT, best_bid, maxLong))
			self.result[STARFRUIT].extend(orders)
			return
		if best_bid > fair_price:
			orders.append(Order(STARFRUIT, best_ask, maxShort))
			self.result[STARFRUIT].extend(orders)
			return

	def tradeORCHIDS(self):
		'''
		HUMIDIY: 60-80% is the ideal range->if it goes above or bellow-> Production decreases 2% for every 5% humidity change
		SUNLIGHT: Has to be >7hours/day -> if not then Production decreases -4%/10 minutes
		max position should be 100
		If the price of buying from southc is less than importing and selling then calculate maxq and do that
		Timestamp: each 100 increment is 1 min of the day so increments from iterations are 1 min
		'''
		ORCHIDS = "ORCHIDS"
		# -------------------------------- Orchid Data ------------------------------- #
		data_key = self.PRODUCTS[ORCHIDS]
		humidity_history = self.DATA[data_key][3]
		humidity_mean = np.mean(humidity_history)
		sunlight_history = self.DATA[data_key][2]
		# ------------------------ Data from the South Island ------------------------ #
		humidity = self.state.observations.conversionObservations[ORCHIDS].humidity
		sunlight = self.state.observations.conversionObservations[ORCHIDS].sunlight
		sun_hours = (sunlight/10000)*24
		#NOTE - keep track of steps with level of sunlight > 7h/day
		if (sun_hours < 7):
			print("sun hours: ",sun_hours)
			self.SUNLIGHT_STEPS += 1
		else:
			self.SUNLIGHT_STEPS = 0
		#NOTE - Calculate humidity
		last_humidity =  humidity_history[int(np.min([99,self.time])-1)]
		humidity_change = (humidity - last_humidity)
		if (humidity > 80 or humidity < 60):
			self.HUMIDITY_CHANGE += humidity_change
		else:
			self.HUMIDITY_CHANGE = 0
		#NOTE - get costs to operate
		exportTariff = self.state.observations.conversionObservations[ORCHIDS].exportTariff
		importTariff = self.state.observations.conversionObservations[ORCHIDS].importTariff
		fees = self.state.observations.conversionObservations[ORCHIDS].transportFees
		south_bid = self.state.observations.conversionObservations[ORCHIDS].bidPrice
		south_ask = self.state.observations.conversionObservations[ORCHIDS].askPrice
		#NOTE - Model the two main factors affecting supply/demmand
		humidity_factor = -0.02 * ((self.HUMIDITY_CHANGE * 100) // 5)
		sunlight_factor = -0.04 * (self.SUNLIGHT_STEPS // 10)
		production_decrease_factor = humidity_factor + sunlight_factor
		production_data = self.DATA[data_key][2]
		production_data[int(np.min([99,self.time]))] = production_decrease_factor
		# -------------------------- BigWhale Island Market -------------------------- #
		bigWhale: OrderDepth = self.state.order_depths[ORCHIDS]
		whale_bid, whale_ask = next(iter(bigWhale.buy_orders)), next(iter(bigWhale.sell_orders))
		whale_midprice = (whale_ask + whale_bid) / 2
		whale_bid_Q, whale_ask_Q = next(iter(bigWhale.buy_orders.values())), next(iter(bigWhale.sell_orders.values()))
		maxShort, maxLong = self.maxOrderSize(ORCHIDS)
		# --------------------- How much does it cost to purchase -------------------- #
		price_to_import = south_ask + fees + importTariff
		price_to_export = whale_bid + fees + exportTariff
		# ----------- There is an arbitrage implicit as we have two markets ---------- #
		orders : List[Order] = []
		if self.time == 99:
			production_data = np.roll(production_data,-1)
		south_supply = south_ask + production_decrease_factor
		# ----------------------------- Arbitrage Entries ---------------------------- #
		PnL_long = 1*whale_bid - 1*south_ask - 1 * importTariff - 1*fees - 1*.1 #Long in my island short in south
		PnL_short = 1*south_bid - 1*whale_ask -1 * exportTariff - 1*fees - 1*.1
		inventory_tolerance = 1
		export_import_balance = 0
		savgol_preds = self.DATA[data_key][1]
		savgol_gradient = np.gradient(savgol_preds)
		# -------------------------- initial buy from island ------------------------- #
		if self.state.position.get(ORCHIDS, 0) < 2:# and savgol_preds[-1] > south_ask:
			stacking_Q = np.max([self.state.position.get(ORCHIDS,0),2]) #low q to minimize position risk and at least 2 to keep one and sell the other one
			orders.append(Order(ORCHIDS, int(south_ask), int(stacking_Q)))
			self.ORCHIDS_INVENTORY_PRICE = ((south_ask*stacking_Q) + np.sum([i for i in self.OPEN_POSITIONS[ORCHIDS].keys()]))/((len(self.OPEN_POSITIONS[ORCHIDS]))+1)
			print(self.ORCHIDS_INVENTORY_PRICE)
			self.result[ORCHIDS].extend(orders)
			return
		# -------------------------- free arbitrage entries -------------------------- #
		if savgol_preds[-1] < 0: #case: our island orchid market > south
			orders.append(Order(ORCHIDS, int(whale_bid), -1))
			self.result[ORCHIDS].extend(orders)
			self.conversions = 1
			return
		if PnL_short > 0:
			orders.append(Order(ORCHIDS, int(whale_bid), 1))
			self.result[ORCHIDS].extend(orders)
			self.conversions = -1
			return

	def tradeBASCKET(self):
		'''
		Take a set of goods and trade them in a basket aka portfolio.
		Objective: optimize the weights of this goods to minimize variance
		from trading data it seems that products from basket tend to revert to the basket price
		'''


	def maxOrderSize(self, product):
		limit = Trader.LIMITS[product]
		productPosition = self.state.position.get(product,0)
		if np.abs(productPosition) == limit:
			return (0, 0)
		shortSize : int = int(-1 * (productPosition + limit))
		longSize : int = int(-1 * (productPosition - limit))
		return (shortSize, longSize)

	def updateData(self): #OK!
		'''
		Method to populate data matrix for keeping track of historic data during execution
		- Features: mid_price, kalman filter, 
		'''
		for i, prod in enumerate(self.PRODUCTS.items()):
			if prod[0] == "ORCHIDS":
				humidity = self.state.observations.conversionObservations["ORCHIDS"].humidity
				sunlight = self.state.observations.conversionObservations["ORCHIDS"].sunlight
				whale_mid_price = Trader.Utils.midPrice(self.state.order_depths[prod[0]])
				south_bid = self.state.observations.conversionObservations["ORCHIDS"].bidPrice
				south_ask = self.state.observations.conversionObservations["ORCHIDS"].askPrice
				south_mid_price = int((south_ask + south_ask)/2)

				time = self.dataTime
				if time < self.program_params["data_length"] - 1:
					self.DATA[i][0][time] = whale_mid_price
					self.DATA[i][4][time] = south_mid_price
					if time > 2:
						preds = Trader.Utils.savitzky_golay(self.DATA[i][0][:time+1])
						self.DATA[i][1][:time+1] = preds
						south_preds = Trader.Utils.savitzky_golay(self.DATA[i][3][:time+1])
						self.DATA[i][1][:time+1] = south_preds
					self.DATA[i][2][time] = sunlight
					self.DATA[i][3][time] = humidity
				if time == self.program_params["data_length"] - 1:
					self.DATA[i][0] = np.roll(self.DATA[i][0],-1)
					self.DATA[i][0][time] = whale_mid_price
					preds = Trader.Utils.savitzky_golay(self.DATA[i][0])
					self.DATA[i][1] = preds
					self.DATA[i][2] = np.roll(self.DATA[i][2],-1)
					self.DATA[i][2] = sunlight
					self.DATA[i][3] = np.roll(self.DATA[i][3],-1)
					self.DATA[i][3] = humidity
					self.DATA[i][4] = np.roll(self.DATA[i][3],-1)
					self.DATA[i][4] = south_mid_price
					south_preds = Trader.Utils.savitzky_golay(self.DATA[i][4])
					self.DATA[i][5] = south_preds

			else:
				mid_price = Trader.Utils.midPrice(self.state.order_depths[prod[0]])
				time = self.dataTime
				if time < self.program_params["data_length"]-1:
					self.DATA[i][0][time] = mid_price
					if time > 2:
						preds = Trader.Utils.savitzky_golay(self.DATA[i][0][:time+1])
						gradient = np.gradient(preds)
						self.DATA[i][1][:time+1] = preds
						self.DATA[i][2][:time+1] = gradient
				if time == self.program_params["data_length"]-1:
					self.DATA[i][0] = np.roll(self.DATA[i][0],-1)
					self.DATA[i][0][time] = mid_price
					preds = Trader.Utils.savitzky_golay(self.DATA[i][0])
					gradient = np.gradient(preds)
					self.DATA[i][1] = preds
					self.DATA[i][2] = gradient
					#after updating midprice we predict for the whole 100 entries

	def calculatePosition(self, takeProfit=0.009, stopLoss=0):
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
			print(close_orders)
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
		self.program_params = {"data_length": 200,}
		# ------------------------ Data Needed for operations ------------------------ #
		self.DATA = np.zeros((7,6,self.program_params["data_length"])) #NOTE - 6 products , 4 data types, 100 entries
		self.OPEN_POSITIONS = {prod:{} for prod in self.PRODUCTS.keys()}
		self.HUMIDITY_CHANGE = 0
		self.SUNLIGHT_STEPS = 0
		self.ORCHIDS_INVENTORY_PRICE = 0
		# -------------- Instance variables to be accessed and modified -------------- #
		self.result : Dict = {prod:[] for prod in self.PRODUCTS.keys()}
		self.state = state
		self.time = int(state.timestamp / 100)
		self.dataTime = np.min([self.time, self.program_params["data_length"]-1])
		self.conversions = 1
		if not self.state.traderData:
			self.state.TraderData = self.DATA
		else:
			self.DATA, self.OPEN_POSITIONS, self.HUMIDITY_CHANGE, self.SUNLIGHT_STEPS, self.ORCHIDS_INVENTORY_PRICE = jp.decode(self.state.traderData, keys=True) #NOTE - OPEN_POSITIONS HAS KEYS AS STR AFTER DECODING
		# -------------------------------- Update Data ------------------------------- #
		self.updateData()
		self.updatePositions()
		self.calculatePosition()
		# --------------------------- 2. Execute New Trades -------------------------- #

		#self.tradeAMETHYSTS()
		self.tradeSTARFRUIT()
		#self.tradeORCHIDS()

		# ------------------------ 3. Optimize Inventory Risk ------------------------ #
		#self.optimizeInventory()
		# ----------------------- 4. Enconde and Return Results ---------------------- #
		data2encode = (self.DATA, self.OPEN_POSITIONS, self.HUMIDITY_CHANGE, self.SUNLIGHT_STEPS, self.ORCHIDS_INVENTORY_PRICE)
		traderData = jp.encode(data2encode, keys=True)
		return self.result, self.conversions, traderData