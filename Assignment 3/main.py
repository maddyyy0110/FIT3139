import numpy as np
from matplotlib import pyplot as plt

def generate_tsp_instance(number_of_cities, grid_size):
    ans = []
    while len(ans) < number_of_cities:
        (x, y) = np.random.randint(0, grid_size, size=2)
        if (x, y) not in ans:
            ans.append((x, y))
    return ans


def plot_instance(instance, grid_size):
    plt.close()
    for i, city in enumerate(instance):
        plt.plot(city[0], city[1], 'o', color='red')
        plt.annotate(i, xy=(city[0], city[1]), xytext=(-8, 8),
        textcoords='offset points')
    plt.grid()
    plt.xlim((-0.5, grid_size+0.5))
    plt.ylim((-0.5, grid_size+0.5))

def plot_tour(instance, grid_size, tour):
    plt.close()
    plot_instance(instance, grid_size)
    for i in range(0, len(tour)-1):
        x = instance[tour[i]][0] 
        y = instance[tour[i]][1] 
        dx = instance[tour[i+1]][0] -x 
        dy = instance[tour[i+1]][1] -y
        plt.arrow(x, y, dx, dy, color='blue')
    x = instance[tour[-1]][0] 
    y = instance[tour[-1]][1] 
    dx = instance[tour[0]][0] -x 
    dy = instance[tour[0]][1] -y
    plt.arrow(x, y, dx, dy, color='blue')
    plt.show()

def time(island1: tuple, island2: tuple, ship_speed: float,current: tuple) -> float:
    x1 = island1[0]
    y1 = island1[1]
    x2 = island2[0]
    y2 = island2[1]

    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    pass

number_of_cities = 20
grid_size = 100
tsp_instance = generate_tsp_instance(number_of_cities, grid_size)

plot_instance(tsp_instance, grid_size)

tour = np.random.permutation(number_of_cities)

plot_tour(tsp_instance,grid_size,tour)