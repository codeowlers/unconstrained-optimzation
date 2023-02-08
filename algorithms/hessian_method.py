import numpy as np
from algorithms import rosenbrock, gradient


# Define the Hessian matrix of the Rosenbrock function
def hessian_rosenbrock(x):
    # Return the Hessian matrix of the Rosenbrock function for input x
    return np.array([[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]],
                     [-400 * x[0], 200]])

def hessian(f, x0, epsilon=1e-4):
    n = len(x0)
    hessian = np.zeros((n, n))
    for i in range(n):
        x_plus = np.copy(x0)
        x_plus[i] += epsilon
        x_minus = np.copy(x0)
        x_minus[i] -= epsilon
        gradient_i = gradient(f, x_plus) - gradient(f, x_minus)
        hessian[i] = gradient_i / (2 * epsilon)
    return hessian
