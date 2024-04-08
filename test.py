from Backtesting. datamodel import Order
import random

orders = {'p1':[], 'p2':[]}
orders2 = {'p1':[]}
for i in range(5):
    
    l = orders['p1']
    l2 = orders['p2']
    l3 = orders2['p1']

    l.append(Order("p1", 10, 10))
    l.append(Order("p1", 7, 10))
    l.append(Order("p1", 9, 10))
    l2.append(Order("p2", 5, 9))
    l3.append(Order("p1", 11, -15))

all_orders = list(orders.values()) + list(orders2.values())
all_orders = sum(all_orders, [])
random.shuffle(all_orders)
LobQueue = {hash(i): i for i in all_orders}

l1 = [[6,7],[8,9,[10,11]]]
l2 = [1,2,3,4,5]
l1 = sum(l1,[])
#print(sum(list(orders.values()),list(orders2.values())))

my_d = {k:{"BUY":{},"SELL":{}} for k in orders.keys()}
for orderObject in all_orders:
    prod = orderObject.symbol
    price = orderObject.price
    side = "BUY" if orderObject.quantity > 0 else "SELL"
    if price not in list(my_d[prod][side].keys()):
         my_d[prod][side][price]  = [hash(orderObject)]
    else:
	    (my_d[prod][side][price]).append(hash(orderObject))

print(my_d)

import jsonpickle

mydata = {
    "T": 1000,
    "t": 100
}

traderdata = jsonpickle.encode(mydata)
print(type(traderdata))
print(traderdata)
data = jsonpickle.decode(traderdata)
print(type(data))
print(data)