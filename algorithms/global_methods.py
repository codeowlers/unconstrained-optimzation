import numpy as np
import time
from algorithms import hessian, gradient, rosenbrock

def backtracking(x, d, f, g, c=0.0001, rho=0.5):
    t = 1
    while f(x + t * d) > f(x) + c * t * np.dot(g, d):
        t *= rho
    return t



# Define the Newton method for finding the minimum of a function
def newton(x0, epsilon=1e-8, max_iter=100):
    # Set the initial value for x as x0
    x = x0
    gradient_rosenbrock = gradient(rosenbrock,x)
    # Set the initial iteration number to 0
    iter = 0
    # Record the start time
    start_time = time.time()
    # Repeat until the gradient norm is less than epsilon or the number of iterations reaches max_iter
    while np.linalg.norm(gradient_rosenbrock) >= epsilon and iter < max_iter:
        # Increase the iteration number by 1
        iter += 1
        # Update x using the Newton method
        hessian_matrix = hessian(rosenbrock, x)
        try:
            inverse = np.linalg.inv(hessian_matrix)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(hessian_matrix)
        x = x - inverse.dot(gradient_rosenbrock)
        gradient_rosenbrock = gradient(rosenbrock,x)
    # Record the end time
    end_time = time.time()
    # Return the optimal point x, the number of iterations, and the elapsed time
    return x, iter, end_time - start_time

# Define the Newton method with backtracking line search for finding the minimum of a function
def newton_back_tracking(x0, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    x = x0
    gradient_x = gradient(rosenbrock, x)
    iter = 0
    start_time = time.time()
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        iter += 1
        hessian_matrix = hessian(rosenbrock, x)
        try:
            inverse = np.linalg.inv(hessian_matrix)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(hessian_matrix)
        dx = -inverse.dot(gradient_x)
        t = backtracking(x, dx, rosenbrock, gradient_x, c, rho)
        x = x + t * dx
        gradient_x = gradient(rosenbrock, x)
    end_time = time.time()
    return x, iter, end_time - start_time


# Define the function `steepest_descent_back_tracking` with five input arguments
def steepest_descent_back_tracking(x0 , epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    x = x0
    gradient_rosenbrock = gradient(rosenbrock, x)
    # Initialize the iteration count to 0
    iter = 0
    # Record the starting time
    start_time = time.time()
    # Loop until either the gradient norm is less than `epsilon` or the number of iterations reaches `max_iter`
    while np.linalg.norm(gradient_rosenbrock) >= epsilon and iter < max_iter:
        # Increase the iteration count by 1
        iter += 1
        # Compute negative gradient 
        dx = -gradient_rosenbrock
        # Update the variable `x` using the gradient and the learning rate
        t = backtracking(x, dx, rosenbrock, gradient_rosenbrock, c, rho)
        x = x + t * dx

    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time


# Define the function `steepest_descent` with four input arguments
def steepest_descent(x0, learning_rate=0.01, epsilon=1e-8, max_iter=100):
    # Initialize the input argument x0 as the variable x
    x = x0
    # Initialize the iteration count to 0
    iter = 0
    # Record the starting time
    start_time = time.time()
    # Loop until either the gradient norm is less than epsilon or the number of iterations reaches max_iter
    while np.linalg.norm(gradient(x)) >= epsilon and iter < max_iter:
        # Increase the iteration count by 1
        iter += 1
        # Update the variable x using the gradient and the learning rate
        x = x - learning_rate * gradient(x)
    # Record the ending time
    end_time = time.time()
    # Return the updated variable x, the number of iterations, and the total time taken
    return x, iter, end_time - start_time