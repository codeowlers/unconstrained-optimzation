import numpy as np
import time

def backtracking(x, d, f, g, c=0.0001, rho=0.5):
    t = 1
    while f(x + t * d) > f(x) + c * t * np.dot(g, d):
        t *= rho
    return t



# Define the Newton method for finding the minimum of a function
def newton(x0, function, gradient, hessian, epsilon=1e-8, max_iter=100):
    # Set the initial value for x as x0
    x = x0
    gradient_x = gradient(x)
    # Set the initial iteration number to 0
    iter = 0
    # Record the start time
    start_time = time.time()
    # Repeat until the gradient norm is less than epsilon or the number of iterations reaches max_iter
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        # Increase the iteration number by 1
        iter += 1
        # Update x using the Newton method
        hessian_x = hessian(x)
        try:
            inverse = np.linalg.inv(hessian_x)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(hessian_x)
        x = x - inverse.dot(gradient_x)
        gradient_x = gradient(x)
    # Record the end time
    end_time = time.time()
    # Return the optimal point x, the number of iterations, and the elapsed time
    return x, iter, end_time - start_time



def newton_back_tracking(x0, function, gradient, hessian, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    x = x0
    x_vals = [x]
    gradient_x = gradient(x)
    iter = 0
    start_time = time.time()
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        iter += 1
        hessian_x = hessian(x)
        try:
            inverse = np.linalg.inv(hessian_x)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(hessian_x)
        dx = -inverse.dot(gradient_x)
        t = backtracking(x, dx, function, gradient_x, c, rho)
        x = x + t * dx
        x_vals.append(x)
        gradient_x = gradient(x)
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
        t = backtracking(x, d, function, gradient_x,c, rho)
        x = x + t * d
        x_vals.append(x)


    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time, x_vals


    # Define the function `steepest_descent` with four input arguments
def steepest_descent(x0, function, gradient, epsilon=1e-8, max_iter=100):
    x = x0
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
        t = 0.0001
        x = x + t * d

    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time


def inexact_newton_back_tracking(x0, function, gradient, hessian, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5, delta=0.5):
    x = x0
    x_vals = [x]
    gradient_x = gradient(x)
    iter = 0
    start_time = time.time()
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        iter += 1
        hessian_x = hessian(x)
        try:
            inverse = np.linalg.inv(hessian_x)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(hessian_x)
        dx = -inverse.dot(gradient_x)
        while True:
            t = backtracking(x, dx, function, gradient_x, c, rho)
            if abs(gradient(x + t * dx).T.dot(dx)) <= delta * np.linalg.norm(gradient_x)**2:
                break
            hessian_x = hessian_x * delta
        x = x + t * dx
        x_vals.append(x)
        gradient_x = gradient(x)
    end_time = time.time()
    return x, iter, end_time - start_time, x_vals
