# Problem 76. Problem 202 in [27]

import numpy as np

'''
    problem76 is a function that calculates the value of the target function for a given point. The target function is
    the mean squared error of the sum of squares of difference between each pair of consecutive elements and the square
    of the first element divided by 10
'''


def problem76(x):
    n = len(x)  # number of elements in x
    Fx = 0  # initialize target function value
    for k in range(n):
        if k < n - 1:  # for elements except the last one
            Fx = Fx + (x[k] - (x[k + 1] ** 2 / 10)) ** 2  # calculate square of the difference
        else:
            Fx = Fx + (x[k] - (x[0] ** 2 / 10)) ** 2  # compare the last element with the square of the first element
    Fx = Fx / 2 # divide the sum by 2
    return Fx


def problem76_gradient(x):
    """
    The function computes the gradient of the Problem 76 optimization problem at a given point x.

    Inputs:
    x: 1-D numpy array of shape (n,) representing the point at which the gradient is to be computed.

    Outputs:
    grad: 1-D numpy array of shape (n,) representing the gradient of the Problem 76 optimization problem at x.
    """
    n = len(x)
    grad = np.zeros(n)
    for k in range(n):
        if k < n - 1:
            grad[k] = 2 * (x[k] - (x[k + 1] ** 2 / 10))
        else:
            grad[k] = 2 * (x[k] - (x[0] ** 2 / 10))
    return grad


def problem76_hessian(x):
    # Compute the Hessian matrix for the given x
    n = len(x)
    # Initialize the hessian matrix with zeros
    hess = np.zeros((n, n))
    # Loop through all elements of the Hessian matrix
    for i in range(n):
        for j in range(n):
            # If i and j are the same, and i < n-1
            if i == j and i < n - 1:
                hess[i, j] = 2
            # If i and j are the same, and i = n-1
            elif i == j and i == n - 1:
                hess[i, j] = 2 + 2 * x[0] / 5
            # If i < n-1 and j = i + 1
            elif i < n - 1 and j == i + 1:
                hess[i, j] = -2 * x[j] / 5
            # If i = n-1 and j = 0
            elif i == n - 1 and j == 0:
                hess[i, j] = -2 * x[j] / 5
    # Return the calculated Hessian matrix
    return hess
