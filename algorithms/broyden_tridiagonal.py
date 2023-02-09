import numpy as np

def broyden_tridiagonal(x, n):
    result = 0
    for k in range(n):
        fk = (3 - 2*x[k]) * x[k] - x[k-1] - 2*x[k+1] + 1
        result += fk ** 2
    return 0.5 * result


def broyden_tridiagonal_gradient(x, n):
    gradient = np.zeros(n)
    for k in range(n):
        fk = (3 - 2*x[k]) * x[k] - x[k-1] - 2*x[k+1] + 1
        gradient[k] = -(3 - 2*x[k]) + x[k-1] + 2*x[k+1]
        gradient[k-1] += -x[k]
        gradient[k+1] += 2*x[k]
    return gradient

def broyden_tridiagonal_hessian(x, n):
    hessian = np.zeros((n, n))
    for k in range(n):
        hessian[k][k] = -2 + 2
        hessian[k-1][k] = 1
        hessian[k][k-1] = 1
        hessian[k+1][k] = 2
        hessian[k][k+1] = 2
    return hessian

