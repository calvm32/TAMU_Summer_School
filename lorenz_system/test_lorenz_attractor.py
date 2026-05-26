import matplotlib.pyplot as plt
import numpy as np

from lorenz_system.solve_ERK import solve_ERK
from lorenz_system.compute_percent_relative_error import compute_percent_relative_error

def test_Lorenz_attractor():
    # Butcher tableau
    R = 4 
    a = [[0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1], [0, 0, 0, 0]] 
    b = [1/6, 1/3, 1/3, 1/6] 
    c = [0, 0.5, 0.5, 0.5] 

    # R = 1 
    # a = [[0]] 
    # b = [1] 
    # c = [0] 

    # Final time
    T = 12

    # Coarse number of time steps
    N = 100*T

    # Lorenz attractor parameters
    sigma = 10
    u_right = 8 / 3
    rho = 28

    # Initial condition
    u_0 = np.array([1.0,1.0,1.0])

    # Right hand side vector
    def F(t,u):
        x = sigma*(u[1] - u[0])
        y = u[0]*(rho - u[2])
        z = u[0]*u[1] - u_right*u[2]
        return np.array([x, y, z]) 

    coarse_ts, coarse_us = solve_ERK(F, u_0, T, N, R, a, b, c) 
    fine_ts, fine_us = solve_ERK(F, u_0, T, N*2, R, a, b, c) 

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot([state[0] for state in coarse_us],[state[1] for state in coarse_us], [state[2] for state in coarse_us], label='coarse solution')
    ax.plot([state[0] for state in fine_us],[state[1] for state in fine_us], [state[2] for state in fine_us], label='fine solution')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title("Lorenz attractor final time {:.2f}".format(fine_ts[-1]))
    plt.legend()

    errors = compute_percent_relative_error(coarse_us, fine_us)
    ax = plt.figure().add_subplot()
    plt.plot(coarse_ts, errors)
    plt.title("Percent relative error over time")
    plt.xlabel("time")
    plt.ylabel("relative error (%)")
    plt.show()

if __name__ == "__main__":
    test_Lorenz_attractor()