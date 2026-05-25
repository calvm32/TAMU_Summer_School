import matplotlib.pyplot as plt
import numpy as np

from projectile_trajectory.find_optimal_angle import find_optimal_angle
from projectile_trajectory.compute_horizontal_distance import compute_horizontal_distance

def test_find_optimal_angle():
    s_0 = 100
    tau = 0.01
    x_0 = 0
    y_0 = 0
    mu = 0.04
    g = 9.80665

    optimal_angle, max_horizontal_distance = find_optimal_angle(s_0, tau, x_0, y_0, mu, g)

    alphas = np.linspace(0,np.pi/2,100)
    xs = [compute_horizontal_distance(alpha, s_0, tau, x_0, y_0, mu, g) for alpha in alphas]
    plt.plot(alphas,xs)

    plt.scatter([optimal_angle],[max_horizontal_distance])

    plt.annotate('optimal launch angle {:.2f} rad\nmax distance {:.2f} m'.format(optimal_angle,max_horizontal_distance),(optimal_angle,max_horizontal_distance), (optimal_angle,max_horizontal_distance - 10))
    plt.xlabel('launch angle (rad)')
    plt.ylabel('horizontal distance traveled (m)')
    plt.title('Max horizontal distance versus launch angle')
    plt.show()

if __name__ == "__main__":
    test_find_optimal_angle()