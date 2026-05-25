import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def solve_wave(a, b, t0, T, c, u_0, v_0, M, N):

    h = (b - a)/(M+1) 
    tau = (T - t0)/(N+1) 

    xs = [a + i * h for i in range(M + 2)]
    ts = [n * tau for n in range(N + 2)]

    # U[i,n] approximates u(x_i, t_n)
    U = np.zeros((M + 2, N + 2))

    # each timestep t_{n+1}, n = 1,...,N, we explicitly update U[:,n+1]

    # initialize t_0
    for i in range(1, M + 1):
        U[i, 0] = u_0(xs[i])
    U[0, 0] = 0
    U[M + 1, 0] = 0

    # initialize t_1
    for i in range(1, M + 1):
        U[i, 1] = U[i, 0] + tau * v_0(xs[i])
    U[0, 1] = 0
    U[M + 1, 1] = 0

    # explicit timestepping
    for n in range(1, N + 1):
        for i in range(1, M + 1):
            U[i, n + 1] = (
                2 * U[i, n]
                - U[i, n - 1]
                + tau**2 * c**2 / h**2 * (U[i + 1, n] - 2 * U[i, n] + U[i - 1, n])
            )
        U[0, n + 1] = 0
        U[M + 1, n + 1] = 0

    return U, xs, ts

if __name__ == "__main__":
    a = -10
    b = 10
    t0 = 0
    T = 40
    c = 1

    u_0 = lambda x: np.exp(-(x**2))
    v_0 = lambda x: 0

    M = 200
    h = (b - a) / (M + 1)
    N = int(c * T / h)
    tau = T / (N + 1)

    U, xs, ts = solve_wave(a, b, t0, T, c, u_0, v_0, M, N)

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