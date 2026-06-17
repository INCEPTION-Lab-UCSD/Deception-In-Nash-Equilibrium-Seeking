import matplotlib.pyplot as plt
import numpy as np

import duopoly
import mutual_deception
import quadratic


def run_duopoly_example():
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

    # anim = duopoly.animate_duopoly_plot(
    #     simulation=simulation,
    #     J_2_ref=J_2_ref,
    #     frame_step=20,
    # )
    # _ = anim

    # duopoly.run_duopoly(
    #     x, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, u, p, m, t
    # )

    # anim = duopoly.animate_reaction_curves(
    #     simulation=simulation,
    #     m=m,
    #     S_d=S_d,
    #     p=p,
    #     isoprofit_levels=[250.0, 500.0, 750.0],
    #     frame_step=20,
    # )
    # anim.animation.save("Duopoly_with_Deception_Simulation.gif", writer="pillow")

    anim = duopoly.duopoly_animation_3d(simulation, m, p, S_d)
    _ = anim

    anim.animation.save("duopoly_game_payoff_curves.gif")
    plt.show()


def run_mutual_deception_duopoly_example():
    x = np.array([50.0, 100.0 / 3.0])
    a = 0.05
    k = -0.03
    # omega_1 = 7877.75
    # omega_2 = 7436.5
    omega_1 = 11877.75
    omega_2 = 12436.5
    J_1_ref = 1200.0
    J_2_ref = 1800.0
    epsilon = -0.001
    epsilon_1 = -0.001
    epsilon_2 = -0.0005
    m = np.array([30.0, 30.0])
    p = 0.2
    S_d = 100.0

    # first_order_simulation = (
    #     mutual_deception.simulate_mutual_deception_duopoly_first_order(
    #         x0=x,
    #         a=a,
    #         k=k,
    #         omega_1=omega_1,
    #         omega_2=omega_2,
    #         J_1_ref=J_1_ref,
    #         J_2_ref=J_2_ref,
    #         epsilon=epsilon,
    #         epsilon_1=epsilon_1,
    #         epsilon_2=epsilon_2,
    #         S_d=S_d,
    #         p=p,
    #         m=m,
    #         horizon=1000.0,
    #         dt=0.001,
    #     )
    # )
    second_order_simulation = (
        mutual_deception.simulate_mutual_deception_duopoly_second_order(
            x0=x,
            a=a,
            k=k,
            omega_1=omega_1,
            omega_2=omega_2,
            J_1_ref=J_1_ref,
            J_2_ref=J_2_ref,
            epsilon=epsilon,
            epsilon_1=epsilon_1,
            epsilon_2=epsilon_2,
            G=np.array([[3.0, 13.0], [2.0, 10.0]]),
            S_d=S_d,
            p=p,
            m=m,
            horizon=50.0,
            dt=0.05,
        )
    )

    # first_order_animation = (
    #     mutual_deception.animate_mutual_deception_duopoly_first_order(
    #         simulation=first_order_simulation,
    #         frame_step=20000,
    #     )
    # )
    second_order_animation = (
        mutual_deception.animate_mutual_deception_duopoly_second_order(
            simulation=second_order_simulation,
            frame_step=3,
        )
    )
    _ = second_order_animation
    ani = second_order_animation.animation
    # ani.save(filename="../first_order_mutual_deception.gif")
    plt.show()


def run_quadratic_example():
    anim = quadratic.run_quadratic_example_6()
    plt.show()


def main():
    run_mutual_deception_duopoly_example()
    # run_quadratic_example()
    # run_duopoly_example()


if __name__ == "__main__":
    main()
