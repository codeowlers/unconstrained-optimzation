import numpy as np
import time


def backtracking(x, d, f, g, c=0.0001, rho=0.5):
    t = 1
    while f(x + t * d) > f(x) + c * t * np.dot(g, d):
        t *= rho
    return t


def newton_back_tracking(x0, function, gradient, hessian, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    x = x0
    x_vals = [x]
    gradient_x = gradient(x)
    iter = 0
    start_time = time.time()
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        hessian_x = hessian(x)
        dx = np.linalg.solve(hessian_x, -gradient_x)
        t = backtracking(x, dx, function, gradient_x, c, rho)
        x = x + t * dx
        x_vals.append(x)
        gradient_x = gradient(x)
        iter += 1
    end_time = time.time()
    return x, iter, end_time - start_time, x_vals


# Define the function `steepest_descent` with four input arguments
def steepest_descent_back_tracking(x0, function, gradient, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    x = x0
    x_vals = [x]
    gradient_x = gradient(x)
    # Initialize the iteration count to 0
    iter = 0
    # Record the starting time
    start_time = time.time()
    # Loop until either the gradient norm is less than `epsilon` or the number of iterations reaches `max_iter`
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        # Increase the iteration count by 1
        iter += 1
        # Compute negative gradient 
        d = -gradient_x
        # Update the variable `x` using the gradient and the learning rate
        t = backtracking(x, d, function, gradient_x, c, rho)
        x = x + t * d
        x_vals.append(x)

    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time, x_vals

    # Define the function `steepest_descent` with four input arguments