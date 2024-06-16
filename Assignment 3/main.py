# Maddy Prazeus, 31494978

import numpy as np
from matplotlib import pyplot as plt
from helper import printProgressBar
import psutil
import multiprocessing
import tqdm

def generate_tsp_instance(number_of_islands, grid_size):
    """Create an array of n randomly placed islands in an m x m grid

    This code was sourced from FIT3139 workshop code named TSP_SA.ipynb

    Args:
        number_of_islands (int): number of islands to create
        grid_size (int): size of grid to place islands on

    Returns:
        Tuple[]: Array of tuples, where each tuple stores a islands x,y coords
    """
    ans = []
    while len(ans) < number_of_islands:
        (x, y) = np.random.randint(0, grid_size, size=2)
        if (x, y) not in ans:
            ans.append((x, y))
    return ans


def plot_instance(instance, grid_size, starting_pos):
    """Plots tsp islands on a grid

    This code was sourced from FIT3139 workshop code named TSP_SA.ipynb

    Args:
        instance (Tuple[]): array of islands to plot
        grid_size (int): size of grid we are creating
        starting_pos: used to highlight one node as the starting position
    """
    # loop over all 
    for i, island in enumerate(instance):
        if i == starting_pos:
            plt.plot(island[0], island[1], 'ko', label = "Start point")
        else:
            plt.plot(island[0], island[1], 'o', color='red')
        plt.annotate(i, xy=(island[0], island[1]), xytext=(-8, 8),
        textcoords='offset points')
    plt.grid()
    plt.xlim((-0.5, grid_size+grid_size/10))
    plt.ylim((-0.5, grid_size+grid_size/10))
    plt.xlabel("x position (km)")
    plt.ylabel("y position (km)")


def plot_tour(instance, grid_size, tour,total_time, current):
    """Plots the path chosen for TSP

    This code was sourced from FIT3139 workshop code named TSP_SA.ipynb

    Args:
        instance (Tuple[]): array of tuples representing cities
        grid_size (int): size of grid to plot
        tour (Tuple[]): Array of island tuples, who's order encodes the path taken
        total_time(float): Total time taken of the tour
        current(Tuple): tuple containg angle and magnitude of current
    """
    # loop over tour and make a connection between each node
    plot_instance(instance, grid_size, tour[0])
    for i in range(0, len(tour)-1):
        x = instance[tour[i]][0] 
        y = instance[tour[i]][1] 
        dx = instance[tour[i+1]][0] -x 
        dy = instance[tour[i+1]][1] -y

        if i == 0:
            arrow_head_length = np.sqrt(dx**2 + dy**2) / 10
            plt.arrow(x, y, dx, dy, color='blue', head_length=arrow_head_length, length_includes_head = True,
                       head_width = arrow_head_length/2, fc='k', ec='k', label = "First move")
        else:
            plt.arrow(x, y, dx, dy, color='blue')

    ## Uncomment if u want to ship to return to start position
    # x = instance[tour[-1]][0] 
    # y = instance[tour[-1]][1] 
    # dx = instance[tour[0]][0] -x 
    # dy = instance[tour[0]][1] -y
    # plt.arrow(x, y, dx, dy, color='blue')
    plt
    plt.title(f"Current Mag: {round(current[0],2)} km/h Current Angle: {round(np.rad2deg(current[1]),2)}°\nTime Taken: {round(total_time,2)} hour(s)")
    plt.legend()

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

    return np.dot(ship_vec,current_vec) / (np.linalg.norm(ship_vec))

def perturb(tour):
    """Given a TSP tour, make a swap between two randomly selected islands

    This code was sourced from FIT3139 workshop code named TSP_SA.ipynb

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
    # extract island x and y vals
    Sx1 = island1[0]
    Sy1 = island1[1]
    Sx2 = island2[0]
    Sy2 = island2[1]

    # create our np path vector and then the velocity vector of the ship
    ship_vec = np.array([Sx2 - Sx1, Sy2 - Sy1])
    distance = np.linalg.norm(ship_vec)
    ship_vel_vec = ship_vec / distance * ship_speed

    # handling if we want to test TSP base model
    if current == None:
        return distance / ship_speed

    # create vector for curent
    Cx1 = Sx1
    Cy1 = Sy1
    Cx2 = Cx1 + current[0] * np.cos(current[1])
    Cy2 = Cy1 + current[0] * np.sin(current[1])

    current_vec = np.array([Cx2 - Cx1, Cy2 - Cy1])

    # get ship speed affected by current
    ship_speed = ship_speed + current_impact(ship_vel_vec,current_vec)

    return distance / ship_speed

def cost(tour,tsp_instance,ship_speed, current):
    """Determine the Cost of a given tour

    Args:
        tour (list): list of integers representing order of islands visited
        tsp_instance (Tuple[]): list of tuples, where each element is an islands x,y coords
        ship_speed (int): velocity of ship
        current (Tuple): Tuple containing currents direction and magnitude

    Returns:
        float: time taken of tour
    """
    total_time = 0

    # loop over each element in tour and add the cost of reaching it
    for i in range(0, len(tour)-1):
        total_time += travel_time(tsp_instance[tour[i]], tsp_instance[tour[i+1]],ship_speed,current)
    ## Uncomment this line if u want ship to return to start position
    #total_time += travel_time(tsp_instance[tour[-1]], tsp_instance[tour[0]],ship_speed,current)
 
    return total_time
    

def SA_TSP(tsp_instance, per_temp, t0, cooling_factor, ship_speed, current):
    """Use simulated annealing to solve a TSP problem

    This code was inspired by FIT3139 workshop code named TSP_SA.ipynb

    Args:
        tsp_instance (Tuple[]): Array of tuples where each tuple corresponds to an island
        per_temp (int): Number of perterbutions to make per iteration
        t0 (int): Initial temperature
        cooling_factor (float): factor used to reduce the temperature
        ship_speed (int): velocity of ship
        current (Tuple): Tuple containing currents direction and magnitude

    Returns:
        Tuple[], float: Array of tuples encoding solution path and time taken of said path
    """


    number_of_cities = len(tsp_instance)
    
    # create random initial solution
    #current_solution = np.random.permutation(number_of_cities) 
    current_solution = range(number_of_cities)
    t = t0
    counter = 0
    
    # loop until temperature below threshold
    while t > 0.001:
        for _ in range(per_temp):
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
    # determine amount of logical cores available to use
    cores = len(psutil.Process().cpu_affinity())
    
    # Initialise all the inputs  for jobs to be done
    items = [0]*n
    currents = []
    for i in range(n):
        #generate current
        current_mag = np.random.uniform(1,SA_TSP_inputs[4]/2)
        current_angle = np.random.uniform(0,2* np.pi)
        currents.append((current_mag,current_angle))

        #add all inputs to items 
        items[i] = [SA_TSP_inputs[0],SA_TSP_inputs[1],
                    SA_TSP_inputs[2],SA_TSP_inputs[3],
                    SA_TSP_inputs[4],(current_mag,current_angle)]

    # create our worker pool
    pool = multiprocessing.Pool(cores)

    # use imap to out of order assign work
    # tqdm is used to create loading bar
    results = []
    for result in tqdm.tqdm(pool.imap(func=starFoo, iterable=items), total=n):
        results.append(result)

    pool.close()
    pool.join()

    return results, currents
 