import numpy as np

# Define the gradient of the Rosenbrock function
def gradient2(x):
    # Return the gradient of the Rosenbrock function for input x
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2)])

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
