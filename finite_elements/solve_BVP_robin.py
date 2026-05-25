import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def exercise_2c():
    N = 100
    alpha = 1.5

    u_0 = lambda x : 0 if x == 0 else 0
    v_0 = lambda x : 1 if x == 1 else 0
    p = lambda x : 1 if x < 1/2 else alpha
    f = lambda x : 1 #np.sin(np.pi*x)

    xs = [i/N for i in range(N+1)]

    def phi(i, x):
        if i > 0 and xs[i-1] <= x <= xs[i]:
            return (x - xs[i-1])/(xs[i] - xs[i-1]) # fixme
        elif i < N and xs[i] <= x <= xs[i+1]:
            return (xs[i+1] - x)/(xs[i+1] - xs[i]) # fixme
        else:
            return 0 # fixme

    # derivative of phi
    def dphi(i,x):
        if i > 0 and xs[i-1] <= x <= xs[i]:
            return 1/(xs[i] - xs[i-1]) # fixme
        elif i < N and xs[i] <= x <= xs[i+1]:
            return -1/(xs[i+1] - xs[i]) # fixme
        else:
            return 0 # fixme

    # system matrix
    A = np.zeros((N+1, N+1))

    # for integration, use quad(integrand, a, b)[0]

    def compute_A(i,j):
        if i == 0 and j == 0:
            integrand = lambda x : p(x)*dphi(0,x)**2
            return quad(integrand, xs[0], xs[1])[0] # fixme
        if i == N and j == N:
            integrand = lambda x : p(x)*dphi(N,x)**2
            return quad(integrand, xs[N-1], xs[N])[0] # fixme
        if i == N and j == N-1:
            integrand = lambda x : p(x)*dphi(i,x)*dphi(j,x)
            return quad(integrand, xs[N-1], xs[N])[0] # fixme

        integrand = lambda x : p(x)*dphi(i,x)*dphi(j,x)
        return quad(integrand, xs[i-1], xs[i+1])[0]

    # right-hand side vector
    F = np.zeros(N+1)

    def compute_F(i):
        integrand = lambda x : f(x)*phi(i,x)
        if i == 0:
            return quad(integrand, xs[0], xs[1])[0] 
        if i == N:
            return quad(integrand, xs[-2], xs[-1])[0] + p(xs[-1])*v_0(xs[-1])
            
        return quad(integrand, xs[i-1], xs[i+1])[0] # fixme

    # equation 0
    A[0,0] = compute_A(0,0)
    F[0] = compute_F(0)

    # equations 1 to N-1
    for i in range(1,N):
        A[i,i-1] = compute_A(i,i-1)
        A[i,i] = compute_A(i,i)
        A[i,i+1] = compute_A(i,i+1)
        F[i] = compute_F(i)

    # equation N
    A[N,N-1] = compute_A(N,N-1)
    A[N,N] = compute_A(N,N)
    F[N] = compute_F(N)

    # solve
    U = np.linalg.solve(A, F)
    U_exact = np.zeros_like(xs)

    C = 1+alpha
    B = 0
    u_exact = lambda x : -1/(2*alpha)*x**2 + (1+1/alpha)*x + B if x >= 1/2 else -(-1/2)*x**2 + C*x
    U_exact[i] = u_exact(xs[i])

    # plot
    plt.plot(xs, U, label="approximate")
    plt.plot(xs, U_exact, label="exact")
    plt.show()

exercise_2c()