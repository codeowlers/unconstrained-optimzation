import numpy as np
from sklearn.metrics import mean_squared_error

from algorithms import steepest_descent_back_tracking, rosenbrock


def grid_search(x0, f, g, learning_rates, epsilons, max_iters, method):
    # Initialize the best parameters and best mean squared error to keep track of the best optimization result
    best_params = None
    best_mse = float('inf')
    # Loop over all possible combinations of learning rates, epsilons, and max_iters
    for learning_rate in learning_rates:
        for epsilon in epsilons:
            for max_iter in max_iters:
                # Call the optimization method (e.g. steepest descent or Newton)
                x, iter, time_elapsed = method(x0, learning_rate, epsilon, max_iter)
                # Calculate the mean squared error between the optimized solution and the true value
                mse = mean_squared_error(x, np.array([1, 1]))
                # Update the best parameters and best mean squared error if the current result is better
                if mse < best_mse:
                    best_mse = mse
                    best_params = (epsilon, max_iter)
    # Return the best optimization parameters and the best mean squared error
    return best_params, best_mse


# Initialize the best result, learning rate, backtracking rate, optimized solution, iteration, elapsed time,
# and the respective best values
def grid_search_steepest_descent(x0):
    # Find the optimal point, number of iterations, and elapsed time using the Newton method
    c_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rho_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

    best_result = float('inf')
    best_c = None
    best_rho = None
    best_x_opt = None
    best_iter = None
    best_elapsed_time = None
    # Loop over all possible combinations of the learning rate and backtracking rate
    for c in c_range:
        for rho in rho_range:
            # Call the steepest descent optimization method with backtracking line search
            x_opt, iter, elapsed_time = steepest_descent_back_tracking(x0, c, rho)
            # Evaluate the result of the optimization method
            result = rosenbrock(x_opt)
            # Update the best parameters and results if the current result is better
            if result < best_result:
                best_result = result
                best_c = c
                best_rho = rho
                best_x_opt = x_opt
                best_iter = iter
                best_elapsed_time = elapsed_time
    # Return the best optimized solution, number of iterations, elapsed time, learning rate, and backtracking rate
    return best_x_opt, best_iter, best_elapsed_time, best_c, best_rho

# Define the range of values for each hyperparameter
# learning_rates = [0.00001, 0.0001, 0.001]
# epsilons = [1e-6, 1e-8, 1e-10]
# max_iters = [100, 500, 1000]

# best_params, best_mse = grid_search(x0, rosenbrock, gradient, learning_rates, epsilons, max_iters)
# print("Best hyperparameters:", best_params)
# print("Best MSE:", best_mse)
