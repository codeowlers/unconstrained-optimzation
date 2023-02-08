def extended_rosenbrock(X):
    def f_k_odd(x, k):
        return 100 * (x[k]**2 - x[k+1])**2

    def f_k_even(x, k):
        return (x[k-1] - 1)**2

    n = len(X)
    Fx = 0
    for i in range(n):
        if i % 2 == 1:
            Fx = Fx + f_k_odd(X, i)
        else:
            Fx = Fx + f_k_even(X, i)
    Fx = Fx / 2
    return Fx


