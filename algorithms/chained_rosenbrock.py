import numpy as np


# Define the function `chained_rosenbrock` that implements the Chained Rosenbrock function
def chained_rosenbrock(x):
    # Get the length of the input vector x
    n = len(x)
    # Initialize the function value
    f = 0
    for i in range(1, n):
        # Update the function value based on the Chained Rosenbrock formula
        f = f + 100 * ((x[i - 1] ** 2 - x[i]) ** 2) + ((x[i - 1] - 1) ** 2)
    # Return the function value
    return f


# Define the function `chained_rosenbrock_gradient` that implements the gradient of the Chained Rosenbrock function
def chained_rosenbrock_gradient(x):
    # Get the length of the input vector x
    n = len(x)
    # Initialize the gradient vector
    gradient = np.zeros(n)
    for i in range(1, n):
        # Update the gradient vector based on the Chained Rosenbrock gradient formula
        gradient[i - 1] = 400 * (x[i - 1] ** 2 - x[i]) * x[i - 1] + 2 * (x[i - 1] - 1)
        gradient[i] = -200 * (x[i - 1] ** 2 - x[i])
    # Return the gradient vector
    return gradient


# Define the function `chained_rosenbrock_hessian` that implements the Hessian matrix of the Chained Rosenbrock function
def chained_rosenbrock_hessian(x):
    # Get the length of the input vector x
    n = len(x)
    # Initialize the Hessian matrix
    hessian = np.zeros((n, n))
    for i in range(1, n):
        # Update the Hessian matrix based on the Chained Rosenbrock Hessian formula
        hessian[i - 1, i - 1] = 400 * (3 * x[i - 1] ** 2 - x[i]) + 2
        hessian[i, i - 1] = hessian[i - 1, i] = -400 * x[i - 1]
    # Return the Hessian matrix
    return hessian

# Define the initial guess for the optimization problem
# n = 10
# x_bar = np.zeros(n)
# for i in range(n):
#     if (i % 2) == 1:
#         x_bar[i] = -1.2
#     else:
#         x_bar[i] = 1.0
