import matplotlib.pyplot as plt
import numpy as np

from projectile_trajectory.compute_trajectory import compute_trajectory
from projectile_trajectory.step_ERK import step_ERK

def test_compute_trajectory():
    alpha = np.pi / 4
    s_0 = 100
    tau = 0.1

    mu = 0.04
    xs, ys = compute_trajectory(alpha,s_0,tau,mu=mu)

    plt.plot(xs, ys)
    plt.title("Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

if __name__ == "__main__":
    test_compute_trajectory()