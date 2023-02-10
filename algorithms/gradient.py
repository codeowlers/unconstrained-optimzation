import numpy as np


def gradient(f, x0, epsilon=1e-4):
    # Initialize the size of gradient to be the same as input x0
    n = len(x0)
    gradient = np.zeros(n)
    for i in range(n):
        # Create a copy of x0 with i-th element incremented by epsilon
        x_plus = np.copy(x0)
        x_plus[i] += epsilon
        # Create a copy of x0 with i-th element decremented by epsilon
        x_minus = np.copy(x0)
        x_minus[i] -= epsilon
        # Approximate the gradient by central difference method
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    # Return the approximate gradient
    return gradient
