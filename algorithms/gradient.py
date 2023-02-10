import numpy as np


def gradient(f, x0, epsilon=1e-4):
    n = len(x0)
    gradient = np.zeros(n)
    for i in range(n):
        x_plus = np.copy(x0)
        x_plus[i] += epsilon
        x_minus = np.copy(x0)
        x_minus[i] -= epsilon
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    return gradient
