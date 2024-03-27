import numpy as np
import math
import matplotlib.pyplot as plt
from helper import bisection, relError


### Task 1.1 cos(x)

def cosTaylor(point,terms):
    """Function to create taylor series approx of cos, given POI and no. of terms

    Args:
        point (float): x value from taylor series
        terms (int): number of terms in taylor series

    Returns:
        float: taylor series approx for a given point
    """
    sum = 0

    for i in range(terms):
        sum += (-1)**i * point**(2*i)/ (math.factorial(2*i))

    return sum


# Code to create graph for 1.1 cos(x)

x = np.linspace(0,5,100)
y = [math.cos(val) for val in x]

plt.plot(x,y,label = "True")

for terms in [1,5,10]:
    y_approx = [cosTaylor(val, terms) for val in x]
    plt.plot(x,y_approx, label = f"{terms} terms")

plt.legend()
plt.title("Taylor series approx of cos(x)")
plt.show()


### Task 1.2 cos(x)

def cosSmallAngle(x):
    return 1 - (x**2/2)


# code to create graph 1.2 cos(x)

x = np.linspace(0,0.25,100)
y = [relError(cosSmallAngle(x_val),math.cos(x_val)) for x_val in x]
plt.plot(x,y)

# function to represent relative error equation
# between small angle approximation of cos(x) and its true value
# note we shift our small angle approx down by 0.01 so we can use root finding methods

function = lambda x: abs((math.cos(x) - (1 - x**2/2))/math.cos(x))*100 - 0.01

root = bisection(function,0.15,0.25,0.001)



plt.plot(root, relError(cosSmallAngle(root),math.cos(root)), marker="o", markersize=10, markeredgecolor="Black", markerfacecolor="Red", label = f"POI x = {root}")
plt.legend()
plt.title("Relative error between small angle approx and cos(x)")
plt.xlabel("x values")
plt.ylabel("Relative Error %")
plt.show()

print(function(root)+0.01)

