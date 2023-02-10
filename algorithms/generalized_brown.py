import numpy as np


def generalized_brown(x):
    n = len(x)
    k = n // 2
    f = 0
    for j in range(k):
        i = 2 * j
        f = f + (x[i - 1] ** 2) ** (x[i] ** 2 + 1) + (x[i] ** 2) ** (x[i - 1] ** 2 + 1)
    return f


# Define the gradient function
def generalized_brown_gradient(x):
    n = len(x)
    grad = np.zeros(n)
    k = n // 2
    for j in range(k):
        i = 2 * j
        grad[i - 1] = 2 * x[i - 1] * (x[i] ** 2 + 1) * (x[i - 1] ** 2) ** (x[i] ** 2) + 2 * x[i] * (
                    x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2)
        grad[i] = 2 * x[i] * (x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2) + 2 * x[i - 1] * (x[i] ** 2 + 1) * (
                    x[i - 1] ** 2) ** (x[i] ** 2)
    return grad


# Define the Hessian function
def generalized_brown_hessian(x):
    n = len(x)
    hess = np.zeros((n, n))
    k = n // 2
    for j in range(k):
        i = 2 * j
        hess[i - 1, i - 1] = 2 * (x[i] ** 2 + 1) * (x[i - 1] ** 2) ** (x[i] ** 2) + 2 * x[i - 1] ** 2 * (
                    x[i] ** 2 + 1) * 2 * x[i] * (x[i - 1] ** 2) ** (x[i] ** 2 - 1)
        hess[i, i - 1] = 2 * x[i - 1] * (x[i - 1] ** 2 + 1) * 2 * x[i] * (x[i] ** 2) ** (x[i - 1] ** 2 - 1)
        hess[i - 1, i] = hess[i, i - 1]
        hess[i, i] = 2 * (x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2) + 2 * x[i] ** 2 * (
                    x[i - 1] ** 2 + 1) * 2 * x[i - 1] * (x[i] ** 2) ** (x[i - 1] ** 2 - 1)
    return hess
