import numpy as np

# Define the gradient of the Rosenbrock function
def gradient(x):
    # Return the gradient of the Rosenbrock function for input x
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2)])