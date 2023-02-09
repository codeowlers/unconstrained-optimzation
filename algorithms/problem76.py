# Problem 76. Problem 202 in [27]

import numpy as np

def problem76(x):
    n = len(x)
    Fx = 0
    for k in range(n):
        if k < n - 1:
            Fx = Fx + (x[k] - (x[k+1]**2 / 10))**2
        else:
            Fx = Fx + (x[k] - (x[0]**2 / 10))**2
    Fx = Fx / 2
    return Fx

def problem76_gradient(x):
    n = len(x)
    grad = np.zeros(n)
    for k in range(n):
        if k < n - 1:
            grad[k] = 2 * (x[k] - (x[k+1]**2 / 10))
        else:
            grad[k] = 2 * (x[k] - (x[0]**2 / 10))
    return grad


# This method uses finite differences to estimate the elements of the hessian matrix
def problem76_hessian(x):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j and i < n - 1:
                hess[i, j] = 2
            elif i == j and i == n - 1:
                hess[i, j] = 2 + 2 * x[0] / 5
            elif i < n - 1 and j == i + 1:
                hess[i, j] = -2 * x[j] / 5
            elif i == n - 1 and j == 0:
                hess[i, j] = -2 * x[j] / 5
    return hess
