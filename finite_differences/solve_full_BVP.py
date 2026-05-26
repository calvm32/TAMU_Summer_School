import numpy as np
import matplotlib.pyplot as plt

"""
solve -p*u_xx + q*u_x + r*u = f
"""

def solve_full_bvp(a, b, p, q, r, f, u_left, u_right, M=100):

    # Assemble the (M + 2) x (M + 2) system matrix A and (M + 2) x 1 right hand
    # side vector F, where A * U = F and U is the (M + 2) x 1 solution vector.
    A = np.zeros((M + 2, M + 2))
    F = np.zeros(M + 2)

    if p != 0: # central difference
        # equations 1 through M
        h = (b-a)/(M+1) 
        for i in range(1, M + 1):
            A[i, i - 1] = -p/h**2 + q/(2*h) 
            A[i, i] = 2*p/h**2 + r(a + h*i) 
            A[i, i + 1] = -p/h**2 - q/(2*h) 
            F[i] = f(a + h*i) 

        # equation 0
        A[0, 0] = 1
        F[0] = u_left

        # equation M + 1
        A[M+1, M+1] = 1
        F[M+1] = u_right

    elif q > 0: # left difference
        # equations 1 through M
        h = (b-a)/(M+1) 
        for i in range(1, M + 1):
            A[i, i - 1] = - q/h 
            A[i, i] = r(a + h*i) + q/h
            A[i, i + 1] = 0 
            F[i] = f(a + h*i) 

        # equation 0
        A[0, 0] = 1
        F[0] = u_left

        # equation M + 1
        A[M+1, M+1] = 1
        F[M+1] = u_right

    else: # right difference
        # equations 1 through M
        h = (b-a)/(M+1) 
        for i in range(1, M + 1):
            A[i, i - 1] = 0
            A[i, i] = r(a + h*i) - q/h
            A[i, i + 1] = q/h 
            F[i] = f(a + h*i) 

        # equation 0
        A[0, 0] = 1
        F[0] = u_left

        # equation M + 1
        A[M+1, M+1] = 1
        F[M+1] = u_right

    # solve A * U = F
    U = np.linalg.solve(A, F)

    return U


if __name__ == "__main__":
    M = 100
    a = 0
    b = 2

    p = 0
    q = 1
    r = lambda x: 1 if x > 1 else 0
    
    f=lambda x: 0 # np.sin(np.pi*x)

    u_left=1
    u_right=0

    # exact solution at the discrete grid points
    h = (b-a)/(M+1) 
    xs = [a + i * h for i in range(M + 2)]

    # approximate solution
    U = solve_full_bvp(a, b, p, q, r, f, u_left, u_right, M)

    # plot solutions
    plt.plot(xs, U, label="approximate")
    ax = plt.gca()
    plt.legend()
    plt.show()