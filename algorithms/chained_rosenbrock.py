import numpy as np


def chained_rosenbrock(x):
    n = len(x)
    f = 0
    for i in range(1, n):
        f = f + 100 * ((x[i - 1] ** 2 - x[i]) ** 2) + ((x[i - 1] - 1) ** 2)
    return f


def chained_rosenbrock_gradient(x):
    n = len(x)
    gradient = np.zeros(n)
    for i in range(1, n):
        gradient[i - 1] = 400 * (x[i - 1] ** 2 - x[i]) * x[i - 1] + 2 * (x[i - 1] - 1)
        gradient[i] = -200 * (x[i - 1] ** 2 - x[i])
    return gradient


def chained_rosenbrock_hessian(x):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(1, n):
        hessian[i - 1, i - 1] = 400 * (3 * x[i - 1] ** 2 - x[i]) + 2
        hessian[i, i - 1] = hessian[i - 1, i] = -400 * x[i - 1]
    return hessian

# Define the initial guess for the optimization problem
# n = 10
# x_bar = np.zeros(n)
# for i in range(n):
#     if (i % 2) == 1:
#         x_bar[i] = -1.2
#     else:
#         x_bar[i] = 1.0
