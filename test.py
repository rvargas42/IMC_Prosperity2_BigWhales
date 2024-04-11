from Backtesting. datamodel import Order
import random
import numpy as np

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

my_data = None

traderdata = jsonpickle.encode(my_data)
print(type(traderdata))
print(traderdata)
data = jsonpickle.decode(traderdata)
print(type(data))
print(data)

def maxOrderSize(position):
	limit = 20
	productPosition = position
	if np.abs(productPosition) > 20:
		return (0, 0)
	shortSize : int = int(-1 * (productPosition + limit))
	longSize : int = int(-1 * (productPosition - limit))
	return (shortSize, longSize)

print(maxOrderSize(-13))


from scipy.optimize import minimize

def objective_function(q, variances):
    return np.dot(variances, q)

def minimize_variance(variances, limits):
    # Convertir las listas en arrays numpy
    variances = np.array(variances)

    # Definir la función de restricción para cada límite
    def constraint(q, limit):
        return limit - np.abs(q)

    # Lista de restricciones de límite para cada variable
    constraints = [{'type': 'ineq', 'fun': constraint, 'args': (limit,)} for limit in limits]

    # Valor inicial para minimizar la varianza
    initial_guess = np.zeros_like(variances)

    # Resolver el problema de optimización
    result = minimize(objective_function, initial_guess, args=(variances,), constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimización fallida")

# Ejemplo de uso
variances = [0.011123342, 0.02335]
limits = [20, 20]
optimal_values = minimize_variance(variances, limits)
print("Valores óptimos:", optimal_values)


# Generar datos de ejemplo
np.random.seed(0)
n = 100  # número de muestras
x1 = np.random.rand(n)  # variable independiente 1
x2 = np.random.rand(n)  # variable independiente 2
x3 = np.random.rand(n)  # variable independiente 3
y = 2 * x1 + 3 * x2 - 5 * x3 + np.random.randn(n)  # variable dependiente

# Construir la matriz de diseño
X = np.column_stack((x1, x2, x3))
X = np.column_stack((np.ones(n), X))  # Agregar columna de unos para el término de sesgo

# Calcular los coeficientes de la regresión lineal
coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

print("Coeficientes de la regresión lineal:")
print(coefficients)

import numpy as np

# Step 1: Prepare your data
X = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 4, 5, 4, 5])  # Dependent variable

# Step 2: Add a bias term
X_bias = np.vstack([X, np.ones(len(X))]).T

# Step 3: Compute the optimal weights (parameters)
weights = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

# Step 4: Make predictions
predictions = X_bias.dot(weights)

# Step 5: Evaluate the model (optional)
mse = np.mean((predictions - y) ** 2)
print("Mean Squared Error:", mse)

# Output the weights
print("Weights (parameters):", weights)