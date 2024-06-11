import numpy as np

def calcAngle(a,b):
    return np.arccos( np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b)))

a = np.array([1,1])
b = np.array([-1,0])

print(np.rad2deg(calcAngle(a,b)))

