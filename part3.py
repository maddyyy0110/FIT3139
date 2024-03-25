import numpy as np
import math
import matplotlib.pyplot as plt

def backwardsError(x):
    yhat = math.cos(2*math.pi*x)
    xhat = math.acos(yhat)/(2*math.pi)
    return abs(xhat - x)

def findSens(forwards, backwards,x,y):
    sens = (forwards/y)/(backwards/x)
    return sens


#cosApprox = lambda x: 1 - 2 * math.pi**2 * x**2

x = np.linspace(0,2,100)
y = [math.cos(2*math.pi*val) for val in x]

plt.plot(x,y,label = "True")

y_approx = [float('%.3g' % val) for val in x]
plt.plot(x,y_approx, label = "approx")


plt.legend()
plt.title("Taylor series approx of cos(2*pi*x)")
plt.show()

forwards = abs(3.5 - math.cos(2*math.pi*3.5))
backwards = backwardsError(3.5)
print(forwards)
print(backwards)


print(findSens(forwards,backwards,3.5,abs(math.cos(2*math.pi*3.5))))