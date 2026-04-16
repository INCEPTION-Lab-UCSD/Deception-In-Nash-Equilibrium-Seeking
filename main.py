import numpy as np

import visualizations


def main():
    x = np.array([50.0, 110.0 / 3.0])
    a = 0.05
    k = 0.03
    omega_1 = 7877.75
    omega_2 = 7436.5
    epsilon = -0.001
    J_2_ref = 1000.0
    u = np.array([0.0, 0.0])
    m = np.array([30.0, 30.0])
    p = 0.2
    S_d = 100.0
    t = 100.0

    visualizations.run_duopoly(
        x, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, u, p, m, t
    )


if __name__ == "__main__":
    main()
