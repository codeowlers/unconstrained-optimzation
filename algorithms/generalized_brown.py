import numpy as np


def generalized_brown(x):
    # Get the length of the input vector x
    n = len(x)
    # Divide the length of x by 2
    k = n // 2
    # Initialize the function value to 0
    f = 0
    for j in range(k):
        # Update the function value based on the Generalized Brown formula
        i = 2 * j
        f = f + (x[i - 1] ** 2) ** (x[i] ** 2 + 1) + (x[i] ** 2) ** (x[i - 1] ** 2 + 1)
    # Return the function value
    return f


# Calculate the gradient of the generalized Brown function
def generalized_brown_gradient(x):
    # Number of variables
    n = len(x)
    # Initialize gradient vector
    grad = np.zeros(n)
    # Number of pairs of variables
    k = n // 2
    for j in range(k):
        # Index of the current pair of variables
        i = 2 * j
        # Calculate gradient for x[i-1]
        grad[i - 1] = 2 * x[i - 1] * (x[i] ** 2 + 1) * (x[i - 1] ** 2) ** (x[i] ** 2) + 2 * x[i] * (
                    x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2)
        # Calculate gradient for x[i]
        grad[i] = 2 * x[i] * (x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2) + 2 * x[i - 1] * (x[i] ** 2 + 1) * (
                    x[i - 1] ** 2) ** (x[i] ** 2)
    # Return the gradient vector
    return grad


# Calculate the hessian of the generalized Brown function
def generalized_brown_hessian(x):
    # n is the length of the input x
    n = len(x)
    # Initialize the hessian matrix with size n x n
    hess = np.zeros((n, n))
    # k is the half of n
    k = n // 2
    # Loop over k, calculate the values of each cell of the hessian matrix
    for j in range(k):

        i = 2 * j
        hess[i - 1, i - 1] = 2 * (x[i] ** 2 + 1) * (x[i - 1] ** 2) ** (x[i] ** 2) + 2 * x[i - 1] ** 2 * (
                    x[i] ** 2 + 1) * 2 * x[i] * (x[i - 1] ** 2) ** (x[i] ** 2 - 1)
        # Hessian matrix is symmetrical, thus its transpose is equal to itself
        hess[i, i - 1] = 2 * x[i - 1] * (x[i - 1] ** 2 + 1) * 2 * x[i] * (x[i] ** 2) ** (x[i - 1] ** 2 - 1)
        hess[i - 1, i] = hess[i, i - 1]
        hess[i, i] = 2 * (x[i - 1] ** 2 + 1) * (x[i] ** 2) ** (x[i - 1] ** 2) + 2 * x[i] ** 2 * (
                    x[i - 1] ** 2 + 1) * 2 * x[i - 1] * (x[i] ** 2) ** (x[i - 1] ** 2 - 1)
    # Return the calculated hessian matrix
    return hess
