import numpy as np
from . import gradient


def hessian(f, x0, epsilon=1e-4):
    # Initialize the size of hessian to be nxn where n is the size of x0
    n = len(x0)
    hessian = np.zeros((n, n))
    for i in range(n):
        # Create a copy of x0 with i-th element incremented by epsilon
        x_plus = np.copy(x0)
        x_plus[i] += epsilon
        # Create a copy of x0 with i-th element decremented by epsilon
        x_minus = np.copy(x0)
        x_minus[i] -= epsilon
        # Calculate the gradient difference between x_plus and x_minus
        gradient_i = gradient(f, x_plus) - gradient(f, x_minus)
        # Approximate the i-th row of the hessian matrix
        hessian[i] = gradient_i / (2 * epsilon)
    # Return the approximate hessian matrix
    return hessian
