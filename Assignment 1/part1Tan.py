import numpy as np
import math
import matplotlib.pyplot as plt
from helper import zagNumsGenerator, bisection, relError




### 1.1 tan(x)

def tanTaylor(point,terms):
    sum = 0

    #generate all the zag numbers we need
    zagNums = zagNumsGenerator(terms)

    #loop over the number of terms, iteratively generating the taylor series approx at our POI
    for i in range(terms):  
        coeff = zagNums[i]
        sum += coeff / math.factorial(2*(i+1)-1) * point**(2*(i+1)-1)

    return sum

# Code to create graph for 1.1 tan(x)

# Using template code from 2024 FIT3139 Applied 1:
x = np.linspace(-1.53,1.53,100)
y = [math.tan(val) for val in x]

plt.plot(x,y,label = "True")

#plotting for varying number of terms in taylor series
for terms in [1,10,50]:
    y_approx = [tanTaylor(val, terms) for val in x]
    plt.plot(x,y_approx, label = f"{terms} terms")

plt.legend()
plt.title("Taylor series approx of tan(x)")
plt.show()




### 1.2 tan(x)


# code to create graph 1.2 tan(x)

x = np.linspace(0,0.025,100)

y = [relError(math.tan(x_val), x_val) for x_val in x]
plt.plot(x,y)


# function to represent relative error equation
# between small angle approximation of tan(x) and its true value
# note we shift our small angle approx down by 0.01 so we can use root finding methods

function = lambda x: abs((math.tan(x) - x)/math.tan(x))*100 - 0.01


root = bisection(function,0.01 , 0.025 ,0.001)


plt.plot(root, relError(math.tan(root),root), marker="o", markersize=10, markeredgecolor="Black", markerfacecolor="Red", label = f"POI x = {root}")
plt.legend()

plt.title("Relative error between small angle approx and tan(x)")
plt.xlabel("x values")
plt.ylabel("Relative Error %")
plt.show()
