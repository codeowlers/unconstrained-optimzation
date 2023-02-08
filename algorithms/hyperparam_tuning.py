import numpy as np
from sklearn.metrics import mean_squared_error

from algorithms import steepest_descent


def grid_search(x0, f, g, learning_rates, epsilons, max_iters):
    best_params = None
    best_mse = float('inf')
    for learning_rate in learning_rates:
        for epsilon in epsilons:
            for max_iter in max_iters:
                x, iter, time_elapsed = steepest_descent(x0, learning_rate, epsilon, max_iter)
                mse = mean_squared_error(x, np.array([1, 1]))  # target is the true value of x
                if mse < best_mse:
                    best_mse = mse
                    best_params = (epsilon, max_iter)
    return best_params, best_mse


# Define the range of values for each hyperparameter
# learning_rates = [0.00001, 0.0001, 0.001]
# epsilons = [1e-6, 1e-8, 1e-10]
# max_iters = [100, 500, 1000]

# best_params, best_mse = grid_search(x0, rosenbrock, gradient, learning_rates, epsilons, max_iters)
# print("Best hyperparameters:", best_params)
# print("Best MSE:", best_mse)
