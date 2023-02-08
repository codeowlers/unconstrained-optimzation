import numpy as np


# Define the Hessian matrix of the Rosenbrock function
def hessian(x):
    # Return the Hessian matrix of the Rosenbrock function for input x
    return np.array([[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]],
                     [-400 * x[0], 200]])
