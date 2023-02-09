import numpy as np

def results_out(method, x_opt, iter, elapsed_time,f):
    print(f"COST of {method}",f(x_opt))
    gradient_value = f(x_opt)
    print("GRADIENT NORM:",np.linalg.norm(gradient_value))
    print(f'Optimal point ({method}): x_opt ={x_opt}')
    print(f'Number of iterations ({method}):{iter}')
    print(f'Elapsed time ({method}): {elapsed_time} seconds \n')
