from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class AggregativeGameSimulation:
    time: np.ndarray
    actions_1: np.ndarray
    actions_2: np.ndarray
    J_1: np.ndarray
    J_2: np.ndarray
    reaction_curves: np.ndarray
    delta: np.ndarray


def J_1_aggregative(x_1, x_2):
    return np.pow(x_1, 4) + np.pow(x_1, 3) + 2.0 * x_1 * x_2


def J_2_aggregative(x_1, x_2):
    return np.exp(x_2) + np.pow(x_2, 2) + 1.1 * x_1 * x_2


def J_2_perceived(x_1, x_2, delta):
    return J_2_aggregative(x_1, x_2) + delta * 0.5 * 1.1 * x_2**2


def J_1_grad_1(x_1, x_2):
    return 4.0 * np.pow(x_1, 3) + 3.0 * np.pow(x_1, 2) + 2.0 * x_2


def J_1_grad_2(x_1):
    return 2 * x_1


def J_2_grad_1(x_2):
    return 1.1 * x_2


def J_2_grad_2(x_1, x_2, delta):
    return np.exp(x_2) + np.pow(x_2, 2) + 1.1 * x_1 + delta * 1.1 * x_2


def delta_update(x_1, x_2, epsilon, J_1_ref):
    return epsilon * (J_1_aggregative(x_1, x_2) - J_1_ref)


def x_i_deceptive(player_idx, deceived_indices, u_i, omega, a, delta, time_value):

    x_i = u_i + a * np.sin(omega[player_idx] * time_value)

    for idx in deceived_indices:
        x_i += a * delta * np.sin(omega[idx] * time_value)

    return x_i


def x_i_oblivious(player_idx, u_i, omega, a, time_value):
    return u_i + a * np.sin(omega[player_idx] * time_value)


def reaction_curve_1(x):
    return -(4 * np.pow(x[0], 3) + 3 * np.pow(x[0], 2)) / 2.0


def simulation_aggregative(
    x0, a, k, omega_1, omega_2, J_1_ref, J_2_ref, epsilon, S_d, p, m, horizon, dt=0.05
):
    time = np.arange(0.0, horizon + dt, dt)
    x = np.asarray(x0, dtype=float)
    actions_1 = np.empty_like(time)
    actions_2 = np.empty_like(time)
    delta = np.empty_like(time)

    reaction_curves = np.empty_like(time)
    omega = np.asarray([omega_1, omega_2])
    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    delta_cur = 0.0
    for idx, time_value in enumerate(time):
        actions_1[idx] = x_i_deceptive(0, [1], x[0], omega, a, delta_cur, time_value)
        actions_2[idx] = x_i_oblivious(1, x[1], omega, a, time_value)
        state = np.asarray([actions_1[idx], actions_2[idx]], dtype=float)
        J_1[idx] = J_1_aggregative(actions_1[idx], actions_2[idx])
        J_2[idx] = J_2_aggregative(actions_1[idx], actions_2[idx])
        reaction_curves[idx] = reaction_curve_1(x)
        delta[idx] = delta_cur

        if idx == len(time) - 1:
            continue

        gradient = np.asarray(
            [
                J_1_grad_1(x[0], x[1]),
                J_2_grad_2(x[0], x[1], delta_cur),
            ]
        )

        x = x - dt * k * gradient
        delta_cur = delta_cur + dt * delta_update(x[0], x[1], epsilon, J_1_ref)

    return AggregativeGameSimulation(
        time=time,
        actions_1=actions_1,
        actions_2=actions_2,
        J_1=J_1,
        J_2=J_2,
        reaction_curves=reaction_curves,
        delta=delta,
    )


def reaction_curve_plot(simulation):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    from scipy.optimize import brentq

    x_1 = np.linspace(-2, 2, 500)

    # Player 1's RC: x_2 = -(4x_1^3 + 3x_1^2) / 2
    rc_1 = -(4 * x_1**3 + 3 * x_1**2) / 2.0

    # Player 2's nominal RC: solve e^x2 + 2*x2 + 1.1*x1 = 0
    def rc_2_nominal(x_1_val, x_2_range=(-10, 10)):
        def eq(x_2):
            return np.exp(x_2) + 2 * x_2 + 1.1 * x_1_val

        try:
            return brentq(eq, *x_2_range)
        except ValueError:
            return np.nan

    # Player 1's deceptive RC: solve e^x2 + 2*(1+0.55*delta)*x2 + 1.1*x1 = 0
    def rc_2_deceptive(x_1_val, delta, x_2_range=(-10, 10)):
        def eq(x_2):
            return np.exp(x_2) + 2 * (1 + 0.55 * delta) * x_2 + 1.1 * x_1_val

        try:
            return brentq(eq, *x_2_range)
        except ValueError:
            return np.nan

    delta_values = np.linspace(-3, 3, 20)
    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=-3, vmax=3)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    # Plot deceptive RCs for Player 2 colored by delta
    for delta in delta_values:
        rc = np.array([rc_2_deceptive(x, delta) for x in x_1])
        valid = np.abs(rc) <= 2
        ax.plot(
            x_1[valid],
            rc[valid],
            color=cmap(norm(delta)),
            linewidth=1.8,
        )

    # Plot Player 1's nominal RC in black
    valid_rc1 = np.abs(rc_1) <= 2
    ax.plot(
        x_1[valid_rc1],
        rc_1[valid_rc1],
        color="black",
        linewidth=2.2,
        label="RC for player 1",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\delta$")
    cbar.set_ticks([-3, -1.5, 0, 1.5, 3])
    cbar.set_ticklabels(
        [
            r"$\delta=-3$",
            r"$\delta=-1.5$",
            r"$\delta=0$",
            r"$\delta=1.5$",
            r"$\delta=3$",
        ]
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0)
    ax.legend(
        loc="upper right", frameon=True, fancybox=False, edgecolor="0.6", fontsize=11
    )

    return fig, ax
