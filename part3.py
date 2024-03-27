import numpy as np
import math
import matplotlib.pyplot as plt


def findCN(forwards, backwards,y,x):
    """Calculate Condition number of a function and its approximation

    Args:
        forwards (float): Forwards error
        backwards (float): Backwards error
        y (float): y value respective to error
        x (float): x value respective to error

    Returns:
        float: Condition number
    """
    sens = abs(forwards/y)/abs(backwards/x)
    return sens



#Calculate CN from forwards and backwards error accross our domain

CNs = []

x = np.linspace(-5,5,100)
y = [math.cos(2*math.pi*val) for val in x]

for i in range(len(x)):
    forwards = abs(1 - math.cos(2*math.pi*x[i]))
    backwards = abs(1-x[i])
    CNs.append(findCN(forwards,backwards,y[i],x[i]))

#plot CN

plt.plot(x,CNs)
plt.axhline(y=1, color='red', linestyle='-')
plt.axis([-5,5  ,-1,5])
plt.title("Graph of CN from forwards backwards error formula")
plt.xlabel("x")
plt.ylabel("CN")
plt.show()


#Plot our taylor series approximation agains the true value.

x = np.linspace(-1,1,100)
y = [math.cos(2*math.pi*val) for val in x]

plt.plot(x,y,label = "True")

y_approx = [float('%.3g' % 1) for _ in x]
plt.plot(x,y_approx, label = "approx")


plt.legend()
plt.title("Taylor series approx of cos(2*pi*x)")
plt.show()



