"""
Author: ravargas.42t@gmail.com
Backtest.py (c) 2024
Desc: Script that simulates market trades between bots
Created:  2024-03-06T17:39:08.622Z
Modified: !date!
"""
from Bot import Bot
from datamodel import TradingState, Order, Listing, OrderDepth, Trade, Time, Symbol, Product, Position, UserId, ObservationValue
import os, sys
import importlib
from typing import Dict, List
import time
import numpy as np
import random

sys.path.append('.')

ID = int

class Config:

	initial_price = 10

	#this is the initial trading state to start simulations
	starting_state: TradingState = TradingState(
		traderData="",
		timestamp= Time,
		listings= {
			"PRODUCT1": Listing(symbol="PRODUCT1", product="PRODUCT1", denomination= "SEASHELLS"),
			"PRODUCT2": Listing(symbol="PRODUCT2", product="PRODUCT2", denomination= "SEASHELLS"),
		},
		order_depths = {
			"PRODUCT1": None,
			"PRODUCT2": None,
		},
		own_trades = {"PRODUCT1": [],"PRODUCT2": []},
		market_trades = {
			"PRODUCT1": [Trade(symbol="PRODUCT1",price=11,quantity=4,buyer="",seller="",timestamp=900)],
			"PRODUCT2": []
		},
		position = {"PRODUCT1": 3,"PRODUCT2": -5},
		observations={}
	)
	max_bots: int = 2

class Backtest(Config):
	'''
	This class will run a backtest over a specified script
	performs operations to return and calculate a credible state
	'''

	def __init__(self,name) -> None:
		self.mid_price = 0
		self.execution_times : List[float] = []
		self.algo_PNL : List[float] = []
		self.benchmark: List[float] = []
		self.state : TradingState = self.starting_state
		self.listings = list(self.state.listings.keys())
		self.Algo_orders: List[Order] = None #list of order objects
		self.Market_orders: List[Order] = None #list of order objects
		self.All_Orders: List[Order] = None
		self.Current_Order: Order = None #this is the hash of the current order being matched
		self.Current_Order_Side: str = None
		self.OrderBookStructure = {k:{"BUY":{},"SELL":{}} for k in self.listings} #searchable structure to match orders and edit matched objects
		self.Matched = []

		try:
			module_name = f'Algos.{name[:-3] if name.endswith(".py") else name}'
			print("running algo: ", module_name)
			trader_module = importlib.import_module(module_name)
			trader_class = getattr(trader_module, "Trader")
			self.trader_instance = trader_class()

		except ImportError as e:
			print(f"Error while trying to import the module {module_name}: {e}")
		except AttributeError as e:
			print(f"Error while getting the class Trader, be sure to name the class trader in {module_name}: {e}")

	def RunMyAlgo(self) -> List[Order]:
		my_algo = self.trader_instance
		start_time = time.perf_counter()
		algo_results = my_algo.run(self.state)[0] #returns just the results dictionary
		end_time = time.perf_counter()
		execution_time = (end_time - start_time) * 1000 #in ms
		self.execution_times.append(execution_time) #adds execution time of algo to class attrib
		orders = list(algo_results.values())
		orders = sum(orders, []) # [order(), order(), ...]
		self.Algo_orders = orders
	
	def IsMatchable(self):
		'''
		Checks if the order can be matched
		'''

	def GetBestBidAsk(self):
		'''
		Takes the Queue and extracts the best bid and ask id
		'''
		self.Best_Bid, self.Best_Ask = None, None
		symbol = self.Current_Order.symbol
		#Get best bid:
		for price_level in reversed(list(self.OrderBookStructure[symbol]["BUY"].keys())):
			price_level_list = self.OrderBookStructure[symbol]["BUY"][price_level]
			if price_level_list:
				self.Best_Bid = price_level_list[0] #first object_id at best bid price level
				break
		for price_level in list(self.OrderBookStructure[symbol]["SELL"].keys()):
			price_level_list = self.OrderBookStructure[symbol]["SELL"][price_level]
			if price_level_list:
				self.Best_Ask = price_level_list[0] #first object_id at best ask price level
				break

	def BuildOrderDepth(self):
		self.OrderDepth = {p: OrderDepth() for p in self.state.listings.keys()}
		for p in self.state.listings.keys():
			top3_Buy, top3_Sell = list(self.OrderBookStructure[p]["BUY"].keys())[-3:], list(self.OrderBookStructure[p]["SELL"].keys())[:3]
			buy_orders, sell_orders = {}, {}
			for level in top3_Buy:
				for element in self.OrderBookStructure[p]["BUY"][level]:
					buy_orders[level] += element.quantity
			for level in top3_Sell:
				for element in self.OrderBookStructure[p]["BUY"][level]:
					sell_orders[level] += element.quantity
			self.OrderDepth[p].buy_orders = buy_orders
			self.OrderDepth[p].sell_orders = sell_orders
		self.state.order_depths = self.OrderDepth

	def SendOrder2End(self):
		'''
		Util function to edit the self.All_Orders list orders to the end to be able to regen a new ob structure with new queues for each level
		'''
		#we send self.Current_Order and self.BestBidOrder/self.BestAskOrder
		if self.side == "BUY": #to distinguish between bestbidorder bestaskorder and send back the correct object
			append_objects = [self.BestAskOrder, self.Current_Order]
			for orderObject in append_objects:
				if orderObject.quantity == 0:
					self.All_Orders.remove(orderObject)
				else:
					self.All_Orders.remove(orderObject)
					self.All_Orders.append(orderObject)
		if self.side == "SELL":
			append_objects = [self.BestBidOrder, self.Current_Order] #get objects that correspond to given hash
			for orderObject in append_objects:
				if orderObject.quantity == 0:
					self.All_Orders.remove(orderObject)
				else:
					self.All_Orders.remove(orderObject)
					self.All_Orders.append(orderObject)

	def MatchOrderBook(self):
		
		'''
		Entry method to match Order objects. This method will edit queue objects and edit self.Matched bool and rerun OrderBookStruct
		'''
		self.BuildFifoQueue() #this is the reference queue to order objects and edit them
		for order_id, orderObject in self.FiFoQueue.items():
			self.Current_Order = orderObject
			self.GetBestBidAsk() #this is where the program will be able to update state of orderbook so we have to edit the orderbookstructure
			symbol = orderObject.symbol
			self.side = order_side = "BUY" if orderObject.quantity > 0 else "SELL"
			self.matched_side = match_side = "BUY" if order_side == "SELL" else "SELL"
			self.match_side_slot = self.OrderBookStructure[symbol][match_side]
			if orderObject.price in list(self.match_side_slot.keys()): #if price is found in opposite side we edit attributes for object at best bid best ask
				#Filter for cases where orders are market orders. This is the only case where an order matched will be edited and sent back at the price queue
				if order_side == "BUY" and orderObject.price >= self.Best_Ask: #we will match with the best ask object
					#get the best_ask object
					self.BestAskOrder = BestAskOrder = self.FiFoQueue[self.Best_Ask]
					#we exhaust the quantity for both orders
					BestAskOrder.quantity = max(0, orderObject.quantity + BestAskOrder.quantity)
					orderObject.quantity = max(0, orderObject.quantity + BestAskOrder.quantity)
					#update order of referenced objects at self.All_orders
					self.SendOrder2End()
					#Rerun and update the orderbook queues
					self.OrderBookStruct()
				if order_side == "SELL" and orderObject.price <= self.Best_Bid: #we will match with the best ask object
					#get the best_ask object
					self.BestBidOrder = BestBidOrder = self.FiFoQueue[self.Best_Bid]
					#we exhaust the quantity for both orders
					BestBidOrder.quantity = max(0, orderObject.quantity + BestBidOrder.quantity)
					orderObject.quantity = max(0, orderObject.quantity + BestBidOrder.quantity)
					#update order of referenced objects at self.All_orders
					self.SendOrder2End()
					#Rerun and update the orderbook queues
					self.OrderBookStruct()
			else:
				self.Matched = True
				continue

	def OrderBookStruct(self):
		for orderObject in self.All_Orders:
			prod = orderObject.symbol
			price = orderObject.price
			side = "BUY" if orderObject.quantity > 0 else "SELL"
			if price not in list(self.OrderBookStructure[prod][side].keys()):
				self.OrderBookStructure[prod][side][price] = [hash(orderObject)]
			else:
				(self.OrderBookStructure[prod][side][price]).append(hash(orderObject))

	def BuildFifoQueue(self):
		self.FiFoQueue = {hash(order):order for order in self.All_Orders}

	def FIFOMatch(self) -> TradingState:
		'''
		This method matches orders and edits the self.state
		Functioning:
			0. Build Queue, add a hash value and shuffle it.
			1. given an initial queue of Order Objects we start matching orders:
				- when we enter the queue we have to construct take a best bid and ask
				- then after having a reference depth we start the match logic
				- this logic will have to differentiate between takers and makers.
				- given a chronological set of orders:
					- 1st - search for a take or make price
					- 2nd - edit order object to new quantities and send to the back of the queue or erase and append to orders and position
					- 3rd - update the order_depth
					- 4rth - repeat until no matches can be done
					- 5th - if no more matches then proceed to next iteration
			2. if the order is partially filled then the order is edited and sent to the back of the queue
			3. Once all the queue has been processed then we return the last order depth
		'''
		#make initial order objects queue
		self.All_Orders = self.Algo_orders + self.Market_orders
		#Populate the searchable structure to be able to edit queues for each price level.
		self.OrderBookStruct()
		print(self.OrderBookStructure)
		#exit matching function if no Queue is empty or no matches can be made
		if not self.OrderBookStructure:
			return
		while not self.Matched:
			self.MatchOrderBook() #start to match orders until no matches are available
		self.BuildOrderDepth() #enter the structure and extract the next state.order_depths

	def Calculate(self):
		Backtest_results = []
		return Backtest_results

	def GenerateBots(self,N: int) -> List[Bot]:
		bots = []
		for i in range(N):
			bullish = np.random.choice([0,1])
			bearish = 0 if bullish == 1 else 1
			bot = Bot(bullish=bullish,bearish=bearish,random_state=np.random.randint(3,15))
			bots.append(bot)
		return bots
	
	def BotExecution(self) -> List[Order]:
		bots: List[Bot] = self.GenerateBots(self.max_bots)
		bot_results = [] #[[bot 1 orders],[]]
		for bot in bots:
			results = bot.run(self.state) #{P:[()], p1:[order()]}
			orders = list(results.values())
			orders = sum(orders, [])
			bot_results.append(orders) # [(),()]
		print(bot_results)
		bot_results = sum(bot_results,[])
		self.Market_orders = bot_results

	def BuildInitialState(self): #TODO Make a reliable way of generating an initial state
		order_depth = self.starting_state.order_depths
		for p in self.listings:
			order_depth[p] = OrderDepth()
			for price in range(1,5):
				order_depth[p].buy_orders[price] = 1
			for price in range(6,10):
				order_depth[p].sell_orders[price] = -1

	def run(self,N: int) -> Dict[str,int]:
		self.BuildInitialState()
		for i in range(N):
			self.BotExecution()
			#run algo
			self.RunMyAlgo()
			#new state calculated with bot orders and algo orders
			self.FIFOMatch()
		backtest_results = self.Calculate()
		return backtest_results #{"Execution_times_ms": [500, ...], "algo_PNL": [1000, ...], "Benchmark": [500,...], "MaxDrawdown": -10%, "Sharpe":1.5}

if __name__ == '__main__':
	backtester = Backtest("algo_mesoplodon_bowdoini.py")
	backtester.run(10)
	#backtester.BotExecution()