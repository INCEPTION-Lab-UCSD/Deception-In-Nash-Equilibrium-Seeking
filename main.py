import matplotlib.pyplot as plt
import numpy as np

import duopoly


def main():
    x = np.array([50.0, 110.0 / 3.0])
    a = 0.05
    k = 0.03
    omega_1 = 7877.75
    omega_2 = 7436.5
    epsilon = -0.001
    J_2_ref = 750.0
    u = np.array([0.0, 0.0])
    m = np.array([30.0, 30.0])
    p = 0.2
    S_d = 100.0
    t = 100.0

    _ = u
    simulation = duopoly.simulate_duopoly(
        x0=x,
        a=a,
        k=k,
        omega_1=omega_1,
        omega_2=omega_2,
        J_2_ref=J_2_ref,
        epsilon=epsilon,
        S_d=S_d,
        p=p,
        m=m,
        horizon=t,
    )

    anim = duopoly.animate_reaction_curves(
        simulation=simulation,
        m=m,
        S_d=S_d,
        p=p,
        isoprofit_levels=[250.0, 500.0, 750.0],
        frame_step=20,
    )

    _ = anim
    plt.show()


if __name__ == "__main__":
    main()
