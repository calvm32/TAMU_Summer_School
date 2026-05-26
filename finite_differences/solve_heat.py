import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def solve_heat(a, b, t0, T, p, f, u_0, u_left, u_right, M, N):

    h = (b - a)/(M+1) 
    tau = (T - t0)/(N+1) 

    xs = [a + i * h for i in range(M + 2)]
    ts = [n * tau for n in range(N + 2)]

    # U[i,n] approximates u(x_i, t_n)
    U = np.zeros((M + 2, N + 2))

    # each timestep t_{n+1}, n = 0,...,N, we assemble a (M + 2) x (M + 2)
    # matrix-vector system A * V = F, solve for V, and store V in column n+1 of U

    # initialize
    for i in range(1, M + 1):
        U[i, 0] = u_0(xs[i]) 
    U[0, 0] = u_left 
    U[M + 1, 0] = u_right 

    # timestepping
    for n in range(N + 1):
        # assemble matrix-vector system
        A = np.zeros((M + 2, M + 2))
        F = np.zeros(M + 2)

        # equations 1 through M
        for i in range(1, M + 1):
            A[i, i - 1] = -p/(2*h**2) 
            A[i, i] = 1/tau +p/(h**2) 
            A[i, i + 1] = -p/(2*h**2)  
            F[i] = (
                U[i, n]/tau
                + p/(2*h**2)*(U[i + 1, n] - 2*U[i, n] + U[i - 1, n])
                + (f(xs[i], ts[n + 1]) + f(xs[i], ts[n])) / 2
            )

        # equation 0
        A[0, 0] = 1
        F[0] = u_left

        # equation M + 1
        A[M + 1, M + 1] = 1
        F[M + 1] = u_right

        # solve and store in U[:,n+1]
        U[:, n + 1] = np.linalg.solve(A, F)

    return U, xs, ts

if __name__ == "__main__":
    a = 0 
    b = 1 
    t0 = 0
    T = 1 
    p = 1 
    f = lambda x, t: 0 
    u_0 = lambda x: np.sin(np.pi*x) 
    u_left = 0 
    u_right = 0 
    M = 100 
    N = 100 

    U, xs, ts = solve_heat(a, b, t0, T, p, f, u_0, u_left, u_right, M, N)

    # --------------------
    # animate the solution
    # --------------------

    # set the initial plot
    fig, ax = plt.subplots()
    U_max = np.max(U)
    U_min = np.min(U)
    abs_U_max = np.max(np.abs(U))
    y_min = U_min - 0.1 * abs_U_max
    y_max = U_max + 0.1 * abs_U_max
    ax.set(
        ylim=[y_min, y_max], xlabel="x", ylabel="u(x,t)", title="t = {}".format(ts[0])
    )
    line = ax.plot(xs, U[:, 0])[0]

    # update the plot each frame
    def update(frame):
        line.set_ydata(U[:, frame])
        ax.set(title="t = {}".format(ts[frame]))
        return line

    # create the animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=N + 2, interval=30)
    plt.show()