import numpy as np
import math
import matplotlib.pyplot as plt


x = np.linspace(0,10,500)
y = [math.cos(2 * np.pi * val) for val in x]


fig, axes = plt.subplots(2)

i = 0
colour = ["royalblue",'firebrick']
precision = 1

for precision in [1,3]:
    y_approx = [math.cos(2 * round(np.pi,precision) * val)  - math.cos(2 * np.pi * val) for val in x]
    axes[i].plot(x,y_approx, label = f"{precision} sig fig(s)", color = colour[i])
    axes[i].legend()
    axes[i].axhline(y=0, color='grey', linestyle='-')
    i+=1

print(y_approx)


fig.suptitle("Difference between cos(2*pi*x) and true value \n with differing precisions of pi")

plt.show()


