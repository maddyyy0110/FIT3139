import numpy as np
from matplotlib import pyplot as plt
import main



# Shared values
number_of_cities = 15
grid_size = 100
ship_speed = 60
tsp_instance = [(47, 45), (16, 51), (48, 39), (7, 41), (67, 8), (27, 76), (34, 6), (25, 18), (66, 77), (15, 99), (61, 27), (49, 72), (84, 20), (87, 62), (98, 59)]

### Section 1:
# current = (30, 0*np.pi/4)

# solution_tour, total_time = main.SA_TSP(tsp_instance, 100, 100, 0.95,ship_speed,None)
# print(total_time)
# print(solution_tour)

# main.plot_tour(tsp_instance,grid_size,solution_tour,total_time,(0,0))

# plt.show()


### Section 2:
# current = (30, 1*np.pi/4)


# solution_tour, total_time = main.SA_TSP(tsp_instance, 100, 100, 0.95,ship_speed,current)
# print(total_time)
# print(solution_tour)

# main.plot_tour(tsp_instance,grid_size,solution_tour,total_time,current) 

# plt.show()


### Section 3:

# solution_tour, total_time = main.SA_TSP(tsp_instance, 100, 100, 0.95,ship_speed,None)
# print(total_time)
# print(solution_tour)

# main.plot_tour(tsp_instance,grid_size,solution_tour,total_time,(0,0)) 
# plt.show()


### Section 4:
# if __name__ == '__main__':

#     SA_TSP_input = [tsp_instance, 20, 100, 0.90,ship_speed]
#     iters = 50
#     results, currents = main.monte_carlo_fast(SA_TSP_input,iters)

#     mean = 0

#     # uncomment to plot every iteration solution
#     # for i in range(iters):
#     #     solution_tour = results[i][0]
#     #     total_time = results[i][1]
#     #     mean += total_time
#     #     main.plot_tour(tsp_instance,grid_size,solution_tour,total_time,currents[i]) 
#     #     plt.show()
#     # print(f"Sample Mean time taken: {mean/iters} hour(s)")


#     # determine optimal route

#     # initialise optimal route and time
#     est_sol_tour = results[0][0]
#     est_total_time = sum([main.cost(est_sol_tour,tsp_instance,ship_speed,current) for current in currents])/iters

#     # loop over every solution tour and compare it and run it against every generated current.
#     # if on avg it performed better than our currrent optimal tour then update
#     for i in range(iters):
#         local_time = 0
#         local_tour = results[i][0]

#         #loop over all currents
#         for j in range(iters):
#             local_time += main.cost(local_tour,tsp_instance,ship_speed,currents[j])

#         # check if local tour performed better
#         local_time = local_time / iters
#         if local_time < est_total_time:
#             est_sol_tour = local_tour
#             est_total_time = local_time

    

#     # sample true current
#     current_mag = np.random.uniform(1,ship_speed/2)
#     current_angle = np.random.uniform(0,2* np.pi)
#     current = (current_mag,current_angle)

#     # find true optimal and plot
#     true_sol_tour, true_total_time = main.SA_TSP(tsp_instance, 100, 100, 0.99,ship_speed,None)
#     main.plot_tour(tsp_instance,grid_size,true_sol_tour,true_total_time,current) 
#     plt.show()

#     #plot avg optimal tour
#     main.plot_tour(tsp_instance,grid_size,est_sol_tour,est_total_time,current) 
#     plt.show()

#     # print % error
#     print(np.abs(true_total_time - est_total_time) / true_total_time * 100)