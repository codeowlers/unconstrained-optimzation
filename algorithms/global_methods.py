import numpy as np
import time

from algorithms import hessian, gradient


# Define the Newton method for finding the minimum of a function
def newton(x0, epsilon=1e-8, max_iter=100):
    # Set the initial value for x as x0
    x = x0
    # Set the initial iteration number to 0
    iter = 0
    # Record the start time
    start_time = time.time()
    # Repeat until the gradient norm is less than epsilon or the number of iterations reaches max_iter
    while np.linalg.norm(gradient(x)) >= epsilon and iter < max_iter:
        # Increase the iteration number by 1
        iter += 1
        # Update x using the Newton method
        x = x - np.linalg.inv(hessian(x)).dot(gradient(x))
    # Record the end time
    end_time = time.time()
    # Return the optimal point x, the number of iterations, and the elapsed time
    return x, iter, end_time - start_time


# Define the function `steepest_descent` with four input arguments
def steepest_descent(x0, learning_rate=0.01, epsilon=1e-8, max_iter=100):
    # Initialize the input argument `x0` as the variable `x`
    x = x0
    # Initialize the iteration count to 0
    iter = 0
    # Record the starting time
    start_time = time.time()
    # Loop until either the gradient norm is less than `epsilon` or the number of iterations reaches `max_iter`
    while np.linalg.norm(gradient(x)) >= epsilon and iter < max_iter:
        # Increase the iteration count by 1
        iter += 1
        # Update the variable `x` using the gradient and the learning rate
        x = x - learning_rate * gradient(x)
    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time
