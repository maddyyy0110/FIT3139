import numpy as np
from matplotlib import pyplot as plt
from helper import printProgressBar
import psutil
import multiprocessing
import timeit
import tqdm

def generate_tsp_instance(number_of_cities, grid_size):
    """Create an array of n randomly placed cities in an m x m grid

    Args:
        number_of_cities (int): number of cities to create
        grid_size (int): size of grid to place cities on

    Returns:
        Tuple[]: Array of tuples, where each tuple stores a cities x,y coords
    """
    ans = []
    while len(ans) < number_of_cities:
        (x, y) = np.random.randint(0, grid_size, size=2)
        if (x, y) not in ans:
            ans.append((x, y))
    return ans


def plot_instance(instance, grid_size):
    """Plots tsp cities on a grid

    Args:
        instance (Tuple[]): array of cities to plot
        grid_size (int): size of grid we are creating
    """
    plt.close()
    for i, city in enumerate(instance):
        plt.plot(city[0], city[1], 'o', color='red')
        plt.annotate(i, xy=(city[0], city[1]), xytext=(-8, 8),
        textcoords='offset points')
    plt.grid()
    plt.xlim((-0.5, grid_size+0.5))
    plt.ylim((-0.5, grid_size+0.5))


def plot_tour(instance, grid_size, tour):
    """Plots the path chosen for TSP

    Args:
        instance (Tuple[]): array of tuples representing cities
        grid_size (int): size of grid to plot
        tour (Tuple[]): Array of city tuples, who's order encodes the path taken
    """
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

def current_impact(ship_vec, current_vec):
    """
    Given a vector for the ship and current, 
    determine the resulting magnitude of the ship vector.

    Args:
        ship_vec (np array): 2D np array in the form [x,y]
        current_vec (np array): 2D np array in the form [x,y]

    Returns:
        float: resulting magnitude of the ship vector
    """

    return np.linalg.norm(current_vec) * np.dot(ship_vec,current_vec)/(np.linalg.norm(ship_vec) * np.linalg.norm(current_vec))

def perturb(tour):
    """Given a TSP tour, make a swap between two randomly selected islands

    Args:
        tour (Tuple[]): Array of tuples, where each tuple encodes the x,y pos of an island

    Returns:
        Tuple[]: Modified tour
    """
    # choose two cities at random
    i, j = np.random.choice(len(tour), 2, replace=False)
    new_tour = np.copy(tour)
    # swap them
    new_tour[i], new_tour[j] =  new_tour[j], new_tour[i]
    return new_tour


def travel_time(island1: tuple, island2: tuple, ship_speed: float,current: tuple) -> float:
    """Given a ship and a current,
    determine the time taken for the ship to travel between two islands

    Args:
        island1 (tuple): (x,y) position of island 1
        island2 (tuple): (x,y) position of island 2
        ship_speed (float): speed of ship in knots
        current (tuple): tuple in the form (x,y) representing the current's vector

    Returns:
        float: speed in knots
    """

    Sx1 = island1[0]
    Sy1 = island1[1]
    Sx2 = island2[0]
    Sy2 = island2[1]

    ship_vec = np.array([Sx2 - Sx1, Sy2 - Sy1])

    Cx1 = Sx1
    Cy1 = Sy1
    Cx2 = Cx1 + current[0] * np.cos(current[1])
    Cy2 = Cy1 + current[0] * np.sin(current[1])

    current_vec = np.array([Cx2 - Cx1, Cy2 - Cy1])


    distance = np.linalg.norm(ship_vec)

    ship_speed = ship_speed + current_impact(ship_vec,current_vec)

    return distance / ship_speed

def cost(tour,tsp_instance,ship_speed, current):

    total_time = 0

    for i in range(0, len(tour)-1):
        total_time += travel_time(tsp_instance[tour[i]], tsp_instance[tour[i+1]],ship_speed,current)
    total_time += travel_time(tsp_instance[tour[-1]], tsp_instance[tour[0]],ship_speed,current)
    
    return total_time
    

def SA_TSP(tsp_instance, perturbations_per_annealing_sep, t0, cooling_factor, ship_speed, current):
    """Use simulated annealing to solve a TSP problem

    Args:
        tsp_instance (Tuple[]): Array of tuples where each tuple corresponds to an island
        perturbations_per_annealing_sep (int): Number of perterbutions to make per iteration
        t0 (int): Initial temperature
        cooling_factor (float): factor used to reduce the temperature

    Returns:
        Tuple[], float: Array of tuples encoding solution path and time taken of said path
    """


    number_of_cities = len(tsp_instance)
    
    # create random initial solution
    current_solution = np.random.permutation(number_of_cities) 
    t = t0
    counter = 0
    
    # loop until temperature below threshold
    while t > 0.001:
        for _ in range(perturbations_per_annealing_sep):
            counter += 1
            # cost of current solution
            current_time_taken = cost(current_solution,tsp_instance, ship_speed, current)
            
            # create new solution and get its cost
            perturbation = perturb(current_solution)
            perturbation_time_taken = cost(perturbation,tsp_instance, ship_speed, current)
            
            #create our delta and determine if we select optimal or suboptimal solution
            delta = perturbation_time_taken - current_time_taken

            # select perturb if it has shorter time taken
            if delta < 0: 
                current_solution = perturbation
                current_time_taken = perturbation_time_taken

            # still select perturb if it takes longer, iff our random val < exp
            elif np.random.rand() < np.exp(-delta/t): 
                current_solution = perturbation
                current_time_taken = perturbation_time_taken
        t = cooling_factor*t


    return current_solution, cost(current_solution,tsp_instance, ship_speed, current)

def monte_carlo(SA_TSP_inputs: list, n:int):
    """Runs a montecarlo simulation on method SA_TSP n times

    Args:
        SA_TSP_inputs (list): list of inputs for SA_TSP method
        n (int): number of times to run the simulation

    Returns:
        list: original tsp instance, list of each solution, its time taken respective current vector
    """

    tour_sols = []

    # loop over simulation size range
    printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range(n):
        # create random current
        current_mag = np.random.uniform(1,SA_TSP_inputs[4]/2)
        current_angle = np.random.uniform(0,2* np.pi)
        current = (current_mag,current_angle)

        # determine solution with given TSP instance and current
        solution_tour, total_time = SA_TSP(SA_TSP_inputs[0],SA_TSP_inputs[1],
                                           SA_TSP_inputs[2],SA_TSP_inputs[3],
                                           SA_TSP_inputs[4],current)
        
        # add to solution list
        tour_sols.append([current,total_time, solution_tour])
        printProgressBar(i + 1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)

    return SA_TSP_inputs[0], tour_sols

def starFoo(args):
    """Wrapper function for SA_TSP
    unpacks list of inputs and calls SA_TSP

    Args:
        args (list): list of inputs

    Returns:
        list: ouput from SA_TSP
    """
    return SA_TSP(*args)

def monte_carlo_fast(SA_TSP_inputs: list, n:int):
    """Runs a montecarlo simulation on method SA_TSP n times

    uses multiprocessing to speedup

    Args:
        SA_TSP_inputs (list): list of inputs for SA_TSP method
        n (int): number of times to run the simulation

    Returns:
        list: original tsp instance, list of each solution, its time taken respective current vector
    """
    cores = len(psutil.Process().cpu_affinity())
    
    items = [0]*n

    for i in range(n):
        #generate current
        current_mag = np.random.uniform(1,SA_TSP_inputs[4]/2)
        current_angle = np.random.uniform(0,2* np.pi)

        #add all inputs to items 
        items[i] = [SA_TSP_inputs[0],SA_TSP_inputs[1],
                    SA_TSP_inputs[2],SA_TSP_inputs[3],
                    SA_TSP_inputs[4],(current_mag,current_angle)]


    pool = multiprocessing.Pool(cores)


    results = []
    for result in tqdm.tqdm(pool.imap(func=starFoo, iterable=items), total=n):
        results.append(result)


    print(results)

    pool.close()
    pool.join()

    return results
    
if __name__ == '__main__':

    number_of_cities = 20
    grid_size = 100

    np.random.seed(42)

    tsp_instance = generate_tsp_instance(number_of_cities, grid_size)

    # plot_instance(tsp_instance, grid_size)



    ship_vec, current_vec = np.array([1,1]), np.array([-1,1])

    # island1 = (0,0)
    # island2 = (10,10)
    ship_speed = 60
    #current = (15, 5*np.pi/4)
    current = (1, 5*np.pi/4)

    # print(travel_time(island1,island2,ship_speed,current))

    # solution_tour, total_time = SA_TSP(tsp_instance, 10, 100, 0.90,ship_speed,current)

    # plot_tour(tsp_instance, grid_size, solution_tour)

    SA_TSP_input = [tsp_instance, 20, 100, 0.90,ship_speed]

    start = timeit.default_timer()
    print("The start time is :", start)
    print(monte_carlo(SA_TSP_input,50))

    #monte_carlo_fast(SA_TSP_input,50)
    print("The difference of time is :", 
                timeit.default_timer() - start)



    # print individual solution
    # tsp_instance = [(15, 69), (52, 37), (66, 11), (56, 33), (26, 73), (87, 32), (32, 89), (61, 18), (9, 11), (43, 30), (21, 22), (78, 71), (50, 68), (73, 83), (58, 63), (26, 40), (79, 52), (14, 88), (86, 24), (40, 30)]
    # solution_tour = [ 4, 15, 12, 14, 13, 11, 16,  7,  2,  8, 10,  1,  5, 18,  3,  9, 19, 0, 17,  6]
    # plot_tour(tsp_instance, grid_size, solution_tour)
