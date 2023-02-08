import numpy as np
# from sklearn.metrics import mean_squared_error

from algorithms import steepest_descent_back_tracking, rosenbrock


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

    for c in c_range:
        for rho in rho_range:
            x_opt, iter, elapsed_time = steepest_descent_back_tracking(x0, c, rho)
            result = rosenbrock(x_opt)
            if result < best_result:
                best_result = result
                best_c = c
                best_rho = rho
                best_x_opt = x_opt
                best_iter = iter 
                best_elapsed_time = elapsed_time  

    return best_x_opt, best_iter, best_elapsed_time, best_c, best_rho
    
    

# Define the range of values for each hyperparameter
# learning_rates = [0.00001, 0.0001, 0.001]
# epsilons = [1e-6, 1e-8, 1e-10]
# max_iters = [100, 500, 1000]

# best_params, best_mse = grid_search(x0, rosenbrock, gradient, learning_rates, epsilons, max_iters)
# print("Best hyperparameters:", best_params)
# print("Best MSE:", best_mse)
