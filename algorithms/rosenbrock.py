import numpy as np

# Define the Rosenbrock function
def rosenbrock(x):
    # Return the Rosenbrock function value for input x
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

# Define the gradient of the Rosenbrock function
def rosenbrock_gradient(x):
    # Return the gradient of the Rosenbrock function for input x
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2)])

# Define the Hessian matrix of the Rosenbrock function
def rosenbrock_hessian(x):
    # Return the Hessian matrix of the Rosenbrock function for input x
    return np.array([[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]],
                     [-400 * x[0], 200]])