# Problem 1. Chained Rosenbrock function
def chained_rosenbrock(x):
    n = len(x)
    sum = 0
    for i in range(1, n):
        if i % 2 == 1:
            x_i = -1.2
        else:
            x_i = 1.0
        x_prev = x[i-1]
        sum += (100 * (x_prev**2 - x_i)**2 + (x_prev - 1)**2)
    return sum