import numpy as np
from .plot import plot


def results_out(method, x_opt, iter, elapsed_time, function, gradient_f, x_vals, plotGraph=False):
    """
    Print results of an optimization method.

    Parameters:
    method (str): Name of the optimization method.
    x_opt (np.array): Optimal point found by the optimization method.
    iter (int): Number of iterations required to reach the optimal point.
    elapsed_time (float): Time taken by the optimization method to reach the optimal point.
    function (function): Function to be optimized.
    gradient_f (function): Gradient of the function to be optimized.
    x_vals (np.array): Initial points to be used in the optimization method.
    plotGraph (bool, optional): Whether to plot the optimization result. Defaults to False.
    """
    print(f"COST of {method}", function(x_opt))
    gradient_value = gradient_f(x_opt)
    print("GRADIENT NORM:", np.linalg.norm(gradient_value))
    print(f'Optimal point ({method}): x_opt ={x_opt}')
    print(f'Number of iterations ({method}):{iter}')
    print(f'Elapsed time ({method}): {elapsed_time} seconds \n')

    if plotGraph is True:
        function_name = " ".join([w.capitalize() for w in function.__name__.split("_")])
        plot(function, x_vals, f' {method} - {function_name} ')
