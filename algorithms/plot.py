import numpy as np
import matplotlib.pyplot as plt


def plot(function, x_vals, title):
    # Plot the contours of the Chained Rosenbrock function
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = function(np.array([X, Y]))
    plt.contour(X, Y, Z)

    # Plot the trajectory of the points generated by the methods
    x_vals = np.array(x_vals)
    plt.scatter(x_vals[:, 0], x_vals[:, 1], c='red', s=4)
    plt.plot(x_vals[:, 0], x_vals[:, 1], c='red', linewidth=1)

    # Plot the last value in green
    last_val = x_vals[-1, :]
    plt.scatter(last_val[0], last_val[1], c='green', s=12, label='Optimal Solution')

    plt.title(title)
    plt.legend()
    plt.show()
