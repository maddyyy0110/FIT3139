from functools import cache
from math import comb as binomial
import numpy as np

# Code sourced The online integer encyclopedia:
# https://oeis.org/search?q=0%2C1%2C0%2C2%2C0%2C16%2C0%2C272&language=english&go=Search

# this function will generate the integer result from the nth derivative of tan(x) evaluated at 0

@cache
def ptan(n):
    # Recursively calls itself
    return (0 if n % 2 == 0 else -sum(binomial(n, k)*ptan(n-k) if k > 0 else 1 for k in range(0, n+1, 2)))

def A350972(n):
    # Only interested in positive results
    t = ptan(n)
    return -t if t < 0 else t
# Peter Luschny, Jun 06 2022

for i in range(10):
    print(ptan(i))

def bisection(a_function,a,b,tol,max_iter = 1000):
    """Bisection Root finding method, Created by 2024 FIT3139 TA's


    Args:
        a_function (float): function for which we want to find a root of
        a (float): lower bound for root
        b (float): upper bound for root
        tol (float): acceptable proximity to root
        max_iter (int, optional): Fail safe to stop loop if root not in [a,b]. Defaults to 1000.

    Raises:
        ValueError: If maximum iteration are reached

    Returns:
        float: root of function
    """
    # Check that bounding values (a,b) are of different signs
    assert np.sign(a_function(a)) != np.sign(a_function(b))

    
    it = 1
    while it <= max_iter:
        # Make midpoint
        c = (a+b)/2
        # check if distance between upper and lower bound are within tolerance
        if abs(b-a) < tol:
            return c
        
        it = it + 1

        # otherwise move one of the bounds to the midpoint
        if np.sign(a_function(c)) == np.sign(a_function(a)):
            a = c
        else:
            b = c
    
    raise ValueError("Maximum iterations achieved.")


def relError(actual,expected):
    # Compute relative error between two values
    return np.abs((actual - expected)/actual)*100\
    