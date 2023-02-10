import numpy as np
from .plot import plot

def results_out(method, x_opt, iter, elapsed_time, function ,gradient_f, x_vals, plotGraph=False):
    print(f"COST of {method}",function(x_opt))
    gradient_value = gradient_f(x_opt)
    print("GRADIENT NORM:",np.linalg.norm(gradient_value))
    print(f'Optimal point ({method}): x_opt ={x_opt}')
    print(f'Number of iterations ({method}):{iter}')
    print(f'Elapsed time ({method}): {elapsed_time} seconds \n')

    if plotGraph is True:
        function_name = " ".join([w.capitalize() for w in function.__name__.split("_")])
        plot(function, x_vals,f' {method} - {function_name} ')

