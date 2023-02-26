import numpy as np
import time


# Define the function `backtracking` that implements backtracking line search
def backtracking(x, d, f, g, c=0.0001, rho=0.5):
    # Initialize the step size
    t = 1
    # Repeat until the Armijo condition is satisfied
    while f(x + t * d) > f(x) + c * t * np.dot(g, d):
        # Decrease the step size by multiplying it with rho
        t *= rho
    # Return the step size
    return t


# Define the function `newton_back_tracking` that implements Newton's method with backtracking line search
def newton_back_tracking(x0, function, gradient, hessian, epsilon=1e-8, max_iter=100, c=1e-4, rho=0.5):
    # Initialize the current estimate of the solution
    x = x0
    # Keep track of the values of x during iteration
    x_vals = [x]
    # Compute the gradient of the function at x0
    gradient_x = gradient(x)
    # Initialize the iteration counter
    iter = 0
    start_time = time.time()
    # Repeat until either the norm of the gradient is less than epsilon, or the number of iterations reaches max_iter
    while np.linalg.norm(gradient_x) >= epsilon and iter < max_iter:
        # Compute the hessian matrix of the function at x
        hessian_x = hessian(x)
        # Compute the direction of negative gradient
        dx = np.linalg.solve(hessian_x, -gradient_x)
        # Find the step size using backtracking line search
        t = backtracking(x, dx, function, gradient_x, c, rho)
        # Update the current estimate of the solution
        x = x + t * dx
        # Append the updated x to the list of x values
        x_vals.append(x)
        # Compute the gradient of the function at the updated x
        gradient_x = gradient(x)
        iter += 1
    end_time = time.time()
    # Return the final estimate of the solution, the number of iterations, the time taken, and the list of x values
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
        gradient_x = gradient(x)


    # Record the ending time
    end_time = time.time()
    # Return the updated variable `x`, the number of iterations, and the total time taken
    return x, iter, end_time - start_time, x_vals
