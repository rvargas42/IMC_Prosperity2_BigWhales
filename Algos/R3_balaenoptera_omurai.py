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
	FILTERING = {
		"STARFRUIT":11,"AMETHYSTS":11,"ORCHIDS":11, "GIFT_BASKET": 3, "CHOCOLATE":4, "STRAWBERRIES": 20, "ROSES": 11
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
		def savitzky_golay(y, window_size=11, order=1, deriv=0, rate=1):
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
			preds = np.convolve(m[::-1], y, mode='valid')
			return preds

	# --------- Class contianing quoting, filtering and any kind of model -------- #

	def tradeAMETHYSTS(self):
		'''Amethysts seem to be capped between two prices: [9996.5 - 10003.5] which is a spread of 7$SH
		Strategy: Calculate a mean price and sell or buy given a distance to price and OBI
		'''
		orders : List[Order] = []
		AMETHYSTS, data_key = "AMETHYSTS", self.PRODUCTS["STARFRUIT"]
		maxShort, maxLong = self.maxOrderSize(AMETHYSTS)
		order_depth: OrderDepth = self.state.order_depths[AMETHYSTS]
		BookImbalance = self.Utils.OrderBookImbalance(order_depth)
		maxOrderSize = self.maxOrderSize(AMETHYSTS)
		amethysts = self.DATA[data_key]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = self.Utils.midPrice(order_depth)

		reservation_bid, reservation_ask, mean = 9999, 10001, 10000
		spread = best_ask-best_bid
		savgol_filter = amethysts[1]
		if best_ask <= mean:
			orders.append(Order(AMETHYSTS, int(best_ask), maxLong))
			orders.append(Order(AMETHYSTS, 1002, -maxLong))
			self.result[AMETHYSTS].extend(orders)
			return
		if best_bid >= savgol_filter[self.dataTime]:
			orders.append(Order(AMETHYSTS, int(best_ask), maxShort))
			orders.append(Order(AMETHYSTS, 9998, -maxShort))
			self.result[AMETHYSTS].extend(orders)
			return

		self.result[AMETHYSTS].extend(orders)

	def tradeSTARFRUIT(self):
		'''

		'''
		STARFRUIT, data_key = "STARFRUIT", self.PRODUCTS["STARFRUIT"]
		starfruit = self.DATA[data_key]
		maxShort, maxLong = self.maxOrderSize(STARFRUIT)
		order_depth: OrderDepth = self.state.order_depths[STARFRUIT]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		Market_sellQ, Market_buyQ = list(order_depth.sell_orders.values()), list(order_depth.buy_orders.values())
		print("Total market Q (sell/buy): ", sum(Market_sellQ), sum(Market_buyQ))
		mid_price = self.DATA[data_key][0][self.dataTime]
		# --------------------- We want to max our inv by the end -------------------- #
		orders = []
		prices = starfruit[0]
		filt = starfruit[1]
		gradient = starfruit[2]
		
		if self.time < 11:
			return
		
		# if int(-maxShort*0.80) >= Market_buyQ[0] and best_bid - 1 > savgol_filter[self.dataTime]:
		# 	orders.append(Order(STARFRUIT, best_bid, -Market_buyQ[0]-1))
		# 	orders.append(Order(STARFRUIT, best_bid - 2, maxLong))
		# 	self.result[STARFRUIT].extend(orders)
		# 	return
		# if int(-maxLong*0.80) >= Market_sellQ[0] and best_ask + 1 < savgol_filter[self.dataTime]:
		# 	orders.append(Order(STARFRUIT, best_ask, -Market_sellQ[0]+1))
		# 	orders.append(Order(STARFRUIT, best_ask + 2, maxShort))
		# 	self.result[STARFRUIT].extend(orders)
		# 	return
		# if starfruit[0][self.dataTime-1] > best_ask:
		# 	if self.state.position.get(STARFRUIT,0) < 0:
		# 		orders.append(Order(STARFRUIT,best_ask,maxLong))
		# 	self.result[STARFRUIT].extend(orders)
		# else:
		# 	orders.append(Order(STARFRUIT, int(starfruit[0][self.dataTime-1]), maxShort))
		# 	self.result[STARFRUIT].extend(orders)
		# 	return
		spread = best_ask - best_bid
		filt_max, filt_min = int(np.max(filt[-50:])), int(np.min(filt[-50:]))
		print("threshold: ", filt_min, filt_max)
		if self.state.position.get(STARFRUIT,0) == 0:
			if best_ask < filt[self.dataTime] and np.isclose(best_ask,mid_price,atol=1):
				print("LONG: mid_price, ask, bid, filter, gradient: ",mid_price,best_ask, best_bid,filt[self.dataTime], gradient[self.dataTime])
				orders.append(Order(STARFRUIT, int(best_ask), -int(best_ask_Q*0.80)))
				self.result[STARFRUIT].extend(orders)
				return
			if best_bid > filt[self.dataTime] and np.isclose(best_bid,mid_price,atol=1):
				print("SHORT: mid_price, ask, bid, filter, gradient: ",mid_price,best_ask, best_bid,filt[self.dataTime], gradient[self.dataTime])
				orders.append(Order(STARFRUIT, int(best_bid), -int(best_bid_Q*0.80)))
				self.result[STARFRUIT].extend(orders)
				return
		else:
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
		if PnL_long > 0 and self.state.position.get(ORCHIDS, 0) >= 2:
			orders.append(Order(ORCHIDS, int(whale_bid), 1))
			self.result[ORCHIDS].extend(orders)
			self.conversions = -1
			return
		if PnL_short > 0 and self.state.position.get(ORCHIDS, 0) >= 2:
			orders.append(Order(ORCHIDS, int(whale_bid), -1))
			self.result[ORCHIDS].extend(orders)
			self.conversions = 1
			return

	def tradeCHOCOBASKET(self):
		BASKET, CHOCO = "GIFT_BASKET", "CHOCOLATE"
		basket_depth, choco_depth = self.state.order_depths["GIFT_BASKET"], self.state.order_depths["CHOCOLATE"]
		basket_bid, basket_ask = next(iter(basket_depth.buy_orders)), next(iter(basket_depth.sell_orders))
		choco_bid, choco_ask = next(iter(choco_depth.buy_orders)), next(iter(choco_depth.sell_orders))
		maxChocoShort, maxChocoLong = self.maxOrderSize("CHOCOLATE")
		maxBasketShort, maxBasketLong = self.maxOrderSize("GIFT_BASKET")
		basket_key, choco_key = self.PRODUCTS["GIFT_BASKET"], self.PRODUCTS["CHOCOLATE"]
		mp_basket_norm, mp_choco_norm = ((self.DATA[basket_key][0] - np.mean(self.DATA[basket_key][0]))/np.std(self.DATA[basket_key][0])), ((self.DATA[choco_key][0] - np.mean(self.DATA[choco_key][0]))/np.std(self.DATA[choco_key][0]))
		spread = mp_basket_norm[self.dataTime] - mp_choco_norm[self.dataTime]
		basket_orders = []
		choco_orders = []

		print("Pair Spread: ", spread)

		if spread > 0:
				basket_orders.append(Order(BASKET, int(basket_bid),int(10)))
				self.result[BASKET].extend(basket_orders)
				choco_orders.append(Order(CHOCO, int(choco_ask),int(-10)))
				self.result[CHOCO].extend(choco_orders)
		
		if spread < 0:
				basket_orders.append(Order(BASKET, int(basket_ask),int(10)))
				self.result[BASKET].extend(basket_orders)
				choco_orders.append(Order(CHOCO, int(choco_bid),int(-10)))
				self.result[CHOCO].extend(choco_orders)

	
	def tradeBASCKET(self): #TODO - 
		'''
		Take a set of goods and trade them in a basket aka portfolio.
		Objective: optimize the weights of this goods to minimize variance
		from trading data it seems that products from basket tend to revert to the basket price
		'''
		BASKET, data_key, orders = "GIFT_BASKET", self.PRODUCTS["GIFT_BASKET"], []
		order_depth: OrderDepth = self.state.order_depths[BASKET]
		gift_basket = self.DATA[data_key]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = (best_bid + best_ask)/2
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		maxShort, maxLong = self.maxOrderSize(BASKET)
		gradient = gift_basket[2]

		if best_ask < gift_basket[1][self.dataTime] and np.isclose(best_ask,mid_price,atol=1):
			orders.append(Order(BASKET, best_ask, int(maxLong)))
			self.result[BASKET].extend(orders)
			return
		if best_bid > gift_basket[1][self.dataTime] and np.isclose(best_bid,mid_price,atol=1):
			orders.append(Order(BASKET, best_bid, int(maxShort)))
			self.result[BASKET].extend(orders)
			return
	
	def tradeROSES(self): #TODO - 
		'''
		Take a set of goods and trade them in a basket aka portfolio.
		Objective: optimize the weights of this goods to minimize variance
		from trading data it seems that products from basket tend to revert to the basket price
		'''
		ROSES, data_key, orders = "ROSES", self.PRODUCTS["ROSES"], []
		order_depth: OrderDepth = self.state.order_depths[ROSES]
		gift_basket = self.DATA[data_key]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = (best_bid + best_ask)/2
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		maxShort, maxLong = self.maxOrderSize(ROSES)
		gradient = gift_basket[2]

		if best_ask < gift_basket[1][self.dataTime]:
			orders.append(Order(ROSES, best_ask, int(maxLong)))
			self.result[ROSES].extend(orders)
			return
		if best_bid > gift_basket[1][self.dataTime]:
			orders.append(Order(ROSES, best_bid, int(maxShort)))
			self.result[ROSES].extend(orders)
			return

	def tradeCHOCO(self): #TODO - 
		'''
		Take a set of goods and trade them in a basket aka portfolio.
		Objective: optimize the weights of this goods to minimize variance
		from trading data it seems that products from basket tend to revert to the basket price
		'''
		CHOCO, data_key, orders = "CHOCOLATE", self.PRODUCTS["CHOCOLATE"], []
		order_depth: OrderDepth = self.state.order_depths[CHOCO]
		gift_basket = self.DATA[data_key]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = (best_bid + best_ask)/2
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		maxShort, maxLong = self.maxOrderSize(CHOCO)
		gradient = gift_basket[2]

		if best_ask < gift_basket[1][self.dataTime]:
			orders.append(Order(CHOCO, best_ask, int(maxLong)))
			self.result[CHOCO].extend(orders)
			return
		if best_bid > gift_basket[1][self.dataTime]:
			orders.append(Order(CHOCO, best_bid, int(maxShort)))
			self.result[CHOCO].extend(orders)
			return

	def tradeSTROBERR(self): #TODO - 
		'''
		Take a set of goods and trade them in a basket aka portfolio.
		Objective: optimize the weights of this goods to minimize variance
		from trading data it seems that products from basket tend to revert to the basket price
		'''
		STROBERR, data_key, orders = "STRAWBERRIES", self.PRODUCTS["STRAWBERRIES"], []
		order_depth: OrderDepth = self.state.order_depths[STROBERR]
		gift_basket = self.DATA[data_key]
		best_bid, best_ask = next(iter(order_depth.buy_orders)), next(iter(order_depth.sell_orders))
		mid_price = (best_bid + best_ask)/2
		best_bid_Q, best_ask_Q = next(iter(order_depth.buy_orders.values())), next(iter(order_depth.sell_orders.values()))
		maxShort, maxLong = self.maxOrderSize(STROBERR)
		gradient = gift_basket[2]

		if best_ask < gift_basket[1][self.dataTime]:
			orders.append(Order(STROBERR, best_ask, int(maxLong)))
			self.result[STROBERR].extend(orders)
			return
		if best_bid > gift_basket[1][self.dataTime]:
			orders.append(Order(STROBERR, best_bid, int(maxShort)))
			self.result[STROBERR].extend(orders)
			return

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
					if time > 10:
						preds = Trader.Utils.savitzky_golay(self.DATA[i][0][:time+1])
						self.DATA[i][1][:time+1] = preds
						south_preds =Trader.Utils.savitzky_golay(self.DATA[i][3][:time+1])
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
				if time < 199:
					self.DATA[i][0][self.dataTime] = mid_price
					if time > 10:
						preds = Trader.Utils.savitzky_golay(self.DATA[i][0][:self.dataTime+1])
						gradient = np.gradient(preds)
						self.DATA[i][1][:self.dataTime+1] = preds
						self.DATA[i][2][:self.dataTime+1] = gradient
				if time == 199:
					self.DATA[i][0] = np.roll(self.DATA[i][0],-1)
					self.DATA[i][0][time] = mid_price
					preds = Trader.Utils.savitzky_golay(self.DATA[i][0])
					gradient = np.gradient(preds)
					self.DATA[i][1] = preds
					self.DATA[i][2] = gradient
					#after updating midprice we predict for the whole 100 entries

	def calculatePosition(self, takeProfit=0.1, stopLoss=0.1):
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
			market_price = 0
			for price, quant in orders.items():
				market_price = best_ask if quant > 0 else best_bid
				profit = ((market_price - price)/price) if quant >= 0 else ((market_price - price)/market_price)
				returns += profit
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
		self.DATA = np.zeros((7,7,200)) #NOTE - 6 products , 4 data types, 100 entries
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
		#self.calculatePosition()
		# --------------------------- 2. Execute New Trades -------------------------- #
		# self.tradeAMETHYSTS()
		# self.tradeSTARFRUIT()
		# self.tradeORCHIDS()
		# self.tradeBASCKET()
		# self.tradeCHOCO()
		# self.tradeROSES()
		# self.tradeSTROBERR()
		self.tradeCHOCOBASKET()
		# ------------------------ 3. Optimize Inventory Risk ------------------------ #
		#self.optimizeInventory()
		# ----------------------- 4. Enconde and Return Results ---------------------- #
		data2encode = (self.DATA, self.OPEN_POSITIONS, self.HUMIDITY_CHANGE, self.SUNLIGHT_STEPS, self.ORCHIDS_INVENTORY_PRICE)
		traderData = jp.encode(data2encode, keys=True)
		return self.result, self.conversions, traderData