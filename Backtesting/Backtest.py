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

	#this is the initial trading state to start simulations
	starting_state: TradingState = TradingState(
		traderData="",
		timestamp= Time,
		listings= {
			"PRODUCT1": Listing(symbol="PRODUCT1", product="PRODUCT1", denomination= "SEASHELLS"),
			"PRODUCT2": Listing(symbol="PRODUCT2", product="PRODUCT2", denomination= "SEASHELLS"),
		},
		order_depths = {
			"PRODUCT1": OrderDepth(),
			"PRODUCT2": OrderDepth(),
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
	
	def IsQueueMatchable(self):
		'''
		Checks if the queue has orders that can be matched
		'''

	def MatchOrder(self):
		
		'''
		Entry Method to Match Orders.
		
		'''
		productName, O_q, O_price = self.Current_Order #retrive the order object

	def GetBestBidAsk(self):
		'''
		Takes the Queue and makes an order depth book
		'''

	def OrderBookStruct(self): #TODO build this structure
		pass

	def FIFOMatch(self) -> TradingState:
		'''
		This method matches orders and edits the self.state
		Functioning:
			0. Build Queue, add a hash value and shuffle it.
			1. given an initial queue of Order Objects we start matching orders:
			   1.1. we have to consider between market makers and takers:
				- when we enter the queue we have to construct take a best bid and ask
				- then after having a reference depth we start the match logic
				- this logic will have to differentiate between takers and makers.
				- given a chronological set of orders:
					- 1st - search for a take or make price
					- 2nd - edit order object to new quantities and send to the back of the queue or erase and append to orders and position
					- 3rd - update the order_depth
					- 4rth - repeat until no matches can be done
					- 5th - if no more matches then proceed to next iteration
			2. if the order is partially filled then the order is edited and sent to the back of the queue as a market sell/buy
			3. Once all the queue has been processed then we return the new order depth 
		'''

		#make orders queue
		self.All_Orders = self.Algo_orders + self.Market_orders
		#Make a searchabke dict structure with available order objects by product, side and price level
		self.OrderBookStruct()
		#exit matching function if no Queue is empty
		if not self.OrderBookStructure:
			return
		while not self.Matched:
			#start iterating the queue until all orders are processed
			for order_id, orderObject in self.OrderBookStructure.items():
				self.Current_Order = orderObject
				self.Current_Order_Side = "BUY" if orderObject.quantity > 0 else "SELL"
				self.MatchOrder()
			
	def Calculate(self): #TODO: calculate this results and edit instance attributes
		Backtest_results = []
		return Backtest_results


	def GenerateBots(self,N: int) -> List[Bot]:
		bots = []
		for i in range(N):
			bullish = np.random.choice(0,1)
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
		bot_results = sum(bot_results,[])
		self.Market_orders = bot_results  

	def run(self,N: int) -> Dict[str,int]:
		for i in range(N):
			self.BotExecution()
			#run algo
			self.RunMyAlgo()
			#new state calculated with bot orders and algo orders
			self.FIFOMatch()
		backtest_results = self.Calculate()
		return backtest_results #{"Average_Execution_time_ms": [500, ...], "algo_PNL": [1000, ...], "Benchmark": [500,...], "MaxDrawdown": -10%, "Sharpe":1.5}
	

if __name__ == '__main__':

	backtester = Backtest("algo_mesoplodon_bowdoini.py")
	backtester.RunMyAlgo()