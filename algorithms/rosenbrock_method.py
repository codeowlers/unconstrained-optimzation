# Define the Rosenbrock function
def rosenbrock(x):
    # Return the Rosenbrock function value for input x
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2