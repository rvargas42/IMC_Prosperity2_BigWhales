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

import timeit

# Define una función para comparar el tiempo de ejecución de ambos métodos
def compare_methods(dict_size):
    # Crea un diccionario con números enteros como llaves y valores
    test_dict = {i: i for i in range(dict_size)}
    
    # Método 1: list(dict.values())[0]
    method1_time = timeit.timeit(lambda: list(test_dict.values())[0], number=10000)
    
    # Método 2: next(iter(dict))
    method2_time = timeit.timeit(lambda: next(iter(test_dict)), number=10000)
    
    # Imprime los resultados
    print(f"Método 1 (list(dict.values())[0]): {method1_time} segundos")
    print(f"Método 2 (next(iter(dict))): {method2_time} segundos")

# Prueba con un diccionario de tamaño 1000
compare_methods(1000)

import time

# Función para verificar si un diccionario está ordenado
def esta_ordenado(diccionario):
    claves = list(diccionario.keys())
    return claves == sorted(claves) or claves == sorted(claves, reverse=True)

# Diccionario grande no ordenado
diccionario_grande = {str(i): i for i in range(100000)}
print("Diccionario grande creado.")

# Medir el tiempo de ordenamiento y verificación
start_time = time.time()

# Ordenar el diccionario sin usar librerías externas
items_ordenados = sorted(diccionario_grande.items())
diccionario_ordenado = {k: v for k, v in items_ordenados}

# Verificar si el diccionario está ordenado
ordenado = esta_ordenado(diccionario_ordenado)

end_time = time.time()

# Imprimir el tiempo transcurrido y el resultado
print("¿El diccionario está ordenado?", ordenado)
print("Tiempo transcurrido:", end_time - start_time, "segundos.")

import numpy as np

# Crear un array de dimensiones (6, 6, 100)
arr = np.zeros((6, 6, 100000))

# Tamaño de un elemento del array en bytes
tamanio_elemento = arr.itemsize

# Número total de elementos en el array
num_elementos = arr.size

# Tamaño total del array en bytes
tamanio_total = tamanio_elemento * num_elementos

print("Tamaño total del array en bytes:", tamanio_total)