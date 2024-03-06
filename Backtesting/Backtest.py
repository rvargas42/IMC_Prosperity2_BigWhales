"""
Author: ravargas.42t@gmail.com
Backtest.py (c) 2024
Desc: Script that simulates 2000 iterations to test each algo
Created:  2024-03-06T17:39:08.622Z
Modified: !date!
"""

from Bot import Bot
from datamodel import TradingState, Order, Listing, OrderDepth, Trade, Time, Symbol, Product, Position, UserId, ObservationValue
import os, sys
import importlib
from typing import Dict, List
import time

sys.path.append('.')

class Config:

    initial_state: TradingState = TradingState(
        traderData="",
        timestamp= Time,
        listings= {
            "PRODUCT1": Listing(symbol="PRODUCT1", product="PRODUCT1", denomination= "SEASHELLS"),
            "PRODUCT2": Listing(symbol="PRODUCT2", product="PRODUCT2", denomination= "SEASHELLS"),
        },
        order_depths = {
            "PRODUCT1": OrderDepth(buy_orders={10: 7, 9: 5}, sell_orders={11: -4, 12: -8}),
            "PRODUCT2": OrderDepth(buy_orders={142: 3, 141: 5}, sell_orders={144: -5, 145: -8}),
        },
        own_trades = {"PRODUCT1": [],"PRODUCT2": []},
        market_trades = {
            "PRODUCT1": [Trade(symbol="PRODUCT1",price=11,quantity=4,buyer="",seller="",timestamp=900)],
            "PRODUCT2": []
        },
        position = {"PRODUCT1": 3,"PRODUCT2": -5},
        observations={}
    )

class Backtest:
    '''
    This class will run a backtest over a specified script
    performs operations to return and calculate a credible state
    '''
    @staticmethod
    def GetAlgo(name: str):
        try:
            module_name = f'Algos.{name[:-3] if name.endswith(".py") else name}'
            print("running algo: ", module_name)

            trader_module = importlib.import_module(module_name)
            trader_class = getattr(trader_module, "Trader")

            trader_instance = trader_class()
            return trader_instance

        except ImportError as e:
            print(f"Error while trying to import the module {module_name}: {e}")
        except AttributeError as e:
            print(f"Error while getting the class Trader, be sure to name the class trader in {module_name}: {e}")

    @staticmethod
    def RunMyAlgo(state):
         
         my_algo = Backtest.GetAlgo()
         start_time = time.perf_counter()
         results = my_algo.run(state)
         end_time = time.perf_counter()
         execution_time = (end_time - start_time) * 1000 #in ms
         return results, execution_time

    @staticmethod
    def GenerateInitialState():
        state = TradingState(

        )

    @staticmethod
    def MatchOrders(market_orders, algo_orders, state: TradingState) -> TradingState:
        '''This method mathches orders and edits TradingState attributes'''
        pass
    
    @staticmethod
    def Calculate(state):
        pass

    @staticmethod
    def BotExecution(*args: Bot) -> TradingState:
        pass

    def run(N) -> Dict[str,float]:
        state: TradingState = Config.initial_state
        bot1, bot2 = Backtest.GenerateBots()
        for i in range(N):
            #run bots
            bot_results = Backtest.BotExecution(state)
            bot_orders = bot_results[0]
            #run algo
            my_algo_results = Backtest.RunMyAlgo()
            my_algo_orders: List[Order] = my_algo_results[0][0]

            #new state calculated with bot orders and algo orders
            state: TradingState = Backtest.MatchOrders(bot_orders,my_algo_orders,state)

        backtest_results = Backtest.Calculate(state)
        return backtest_results #{"Average_Execution_time_ms": [500, ...], "algo_PNL": [1000, ...], "Benchmark": [500,...], "MaxDrawdown": -10%, "Sharpe":1.5}

def main():

    backtest_results = Backtest.run(2000) #2000 iterations
    print(backtest_results)
    return 1
    

if __name__ == '__main__':
    
    Backtest.GetAlgo("algo_mesoplodon_bowdoini.py")
