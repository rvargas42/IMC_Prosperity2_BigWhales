import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from Utils.Config import CHARTS_DIR


treasure_map = np.array(
    [
        [2,4,3,3,4],
        [3,5,5,5,3],
        [4,5,8,7,2],
        [5,5,5,5,4],
        [2,3,4,2,3],
    ]
)

multiplier = np.array(
    [
        [24, 70, 41, 21, 60],
        [47, 82, 87, 80, 35],
        [73, 89, 100, 90, 17],
        [77, 83, 85, 79, 55],
        [12, 27, 52, 15, 30]
    ]
)

share = 1/(treasure_map+1)

expected_profit =  share * multiplier

#Make simulations;

expeditions = multiplier.flatten()
d = {i: 0 for i in expeditions}
PIRATES = 3000

expeditions = list(expeditions)
initial_pirates = list(treasure_map.flatten())

def greedy_choice(number_of_pirates, locations_map, expeditions):
    total_pirates_proportion = sum(initial_pirates)
    choices = []
    pirates_at_location = locations_map.values()
    for i in range(1,13):
        choice = np.random.choice(expeditions, size=1)
        pick = choice[0]
        share_greed_threshold = np.random.uniform(0.015, 0.075)
        if (locations_map[pick] / number_of_pirates) <= share_greed_threshold and len(choices) < 3:
            c = choice[0]
            locations_map[c] += 1
            choices.append(c)
        if i == 12 and len(choices)==0:
            c = choice[0]
            locations_map[c] += 1
            choices.append(c)
        else:
            continue
    return choices


def run_simulation(Caribean, expeditions):
    from collections import Counter
    #cache of locations chosen to then apply percentage
    locations_chosen = []
    location_choices = []
    profits = []
    number_of_expeditions = []
    profit_map = {k:0 for k in expeditions}
    locations_map = {k:v for k,v in zip(expeditions,initial_pirates)}
    expedition_cost = 25000
    #random chest locations
    chest_locations = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]
    np.random.shuffle(chest_locations)
    chest_prices = [7500 if i == 1 else 0 for i in chest_locations]
    #make a random selection some may be informed and others not
    pirate = 0
    while pirate < Caribean:
        #Plan expeditions
        locations = greedy_choice(pirate, locations_map, expeditions=expeditions)
        number_of_expeditions.append(len(locations))
        locations_chosen.extend(locations)
        location_choices.append(list(locations))
        profit = 0
        n = 0
        for location in locations:
            j = 0
            while expeditions[j] != location:
                j+=1
            if chest_locations[j] == 0:
                continue
            else:
                Loot = chest_prices[j]
                profit += (location)*Loot - (n*expedition_cost)
                profit_map[location] += profit * (locations_map[location]/Caribean)
                profits.append(int(profit))
            n+=1
        pirate+=1
    return profits, locations_map, location_choices, profit_map, number_of_expeditions
            
p, locations_map, choices, profit_map,numberExpeditions = run_simulation(int(input("How many Pirates will sail in search of treasures? ")), expeditions)
# Suponiendo que tienes las listas 'loc', 'profits' y 'occ'
loc = list(locations_map.keys())
occ = numberExpeditions
profits = np.array((list(profit_map.values()))) / np.array(occ)




from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


# Convertir loc y occ a índices numéricos
x = loc

# Crear una cuadrícula para la interpolación
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(occ), max(occ), 100))

# Interpolar los profits en la cuadrícula
z_grid = griddata((x, occ), profits, (x_grid, y_grid), method='cubic')

# Graficar la superficie
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')

# Etiquetar ejes y título
ax.set_xticks(x)
ax.set_xlabel('Location')
ax.set_ylabel('Number of pirates at location')
ax.set_zlabel('Net Profit')
ax.set_title('Treasure Hunt simulation')

# Agregar una barra de color para la superficie
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
fig.subplots_adjust()

# Mostrar el gráfico
plt.savefig(f"{CHARTS_DIR}/MANUAL_profit_simulation.jpg")
plt.show()
