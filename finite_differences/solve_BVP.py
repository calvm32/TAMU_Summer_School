import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(a, b, p, f, alpha, beta, M=100):

    # Assemble the (M + 2) x (M + 2) system matrix A and (M + 2) x 1 right hand
    # side vector F, where A * U = F and U is the (M + 2) x 1 solution vector.
    A = np.zeros((M + 2, M + 2))
    F = np.zeros(M + 2)

    # equations 1 through M
    h = (b-a)/(M+1) 
    for i in range(1, M + 1):
        A[i, i - 1] = -p/h**2
        A[i, i] = 2*p/h**2
        A[i, i + 1] = -p/h**2
        F[i] = f(a + h*i) 

    # equation 0
    A[0, 0] = 1
    F[0] = alpha

    # equation M + 1
    A[M+1, M+1] = 1
    F[M+1] = beta

    # solve A * U = F
    U = np.linalg.solve(A, F)

    return U

def test_bvp(a, b, p, f, alpha, beta, M):

    # exact solution at the discrete grid points
    h = (b-a)/(M+1) 
    xs = [a + i * h for i in range(M + 2)]
    U_exact = [u_exact(x) for x in xs]

    # approximate solution
    U = solve_bvp(a, b, p, f, alpha, beta, M)

    # plot solutions
    plt.plot(xs, U_exact, label="exact")
    plt.plot(xs, U, label="approximate")
    plt.legend()
    plt.show()

def compute_errors(a, b, p, f, alpha, beta):

    # table headers
    print("h\t\tE_M")

    for M in [9, 19, 39, 79, 159]:
        U = solve_bvp(a, b, p, f, alpha, beta, M)

        # exact solution at the discrete grid points
        h = (b-a)/(M+1) 
        xs = [a + i * h for i in range(M + 2)]
        U_exact = [u_exact(x) for x in xs]

        E_M = max([ abs(U[i] - U_exact[i]) for i in range(M + 2)]) 

        # table entry
        print("{:e}\t{:e}".format(h, E_M))

if __name__ == "__main__":
    M=10
    a = 0
    b = 1
    p = 1
    f=lambda x: np.sin(np.pi*x)
    alpha=0
    beta=0

    # exact solution
    def u_exact(x):
        return (1/np.pi)**2*np.sin(np.pi*x) 

    compute_errors(a, b, p, f, alpha, beta)
    test_bvp(a, b, p, f, alpha, beta, M)