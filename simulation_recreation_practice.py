from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from duopoly import *


@dataclass
class DuopolySimulation:
    time: np.ndarray
    actions_1: np.ndarray
    actions_2: np.ndarray
    actions_1_deception: np.ndarray
    actions_2_deception: np.ndarray
    J_1: np.ndarray
    J_2: np.ndarray
    J_1_deception: np.ndarray
    J_2_deception: np.ndarray
    delta: np.ndarray
    x_star: np.ndarray
    x_delta_star: np.ndarray
    delta_star: float


@dataclass
class ReactionCurveAnimation:
    animation: animation.FuncAnimation
    figure: plt.Figure
    axes: plt.Axes


def simulation_duopoly(
    x0, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, p, m, horizon, dt=0.05
):
    time = np.arange(0.0, horizon + dt, dt)

    x_nominal = np.array(x0, dtype=float).copy()
    x_deception = np.array(x0, dtype=float).copy()
    delta_2 = 0.0

    actions_1 = np.empty_like(time)
    actions_2 = np.empty_like(time)
    actions_1_deception = np.empty_like(time)
    actions_2_deception = np.empty_like(time)

    J_1 = np.empty_like(time)
    J_2 = np.empty_like(time)
    J_1_deception = np.empty_like(time)
    J_2_deception = np.empty_like(time)
    delta = np.empty_like(time)

    reaction_curves_deception = np.empty_like(time)

    for idx, time_value in enumerate(time):
        actions_1[idx] = x_1_duopoly(x_nominal[0], a, omega_1, time_value)
        actions_2[idx] = x_2_duopoly(x_nominal[1], a, omega_1, omega_2, 0.0, time_value)

        actions_1_deception[idx] = x_1_duopoly(x_deception[0], a, omega_1, time_value)
        actions_2_deception[idx] = x_2_duopoly(
            x_deception[1], a, omega_1, omega_2, delta_2, time_value
        )

        state_deception = np.array(
            [actions_1_deception[idx], actions_2_deception[idx]], dtype=float
        )

        s_1 = s_1_duopoly(actions_1[idx], actions_2[idx], p, S_d)
        s_2 = s_2_duopoly([actions_1[idx], actions_2[idx]], p, S_d)

        s_2_deception = s_2_duopoly(
            p, actions_1_deception[idx], actions_2_deception[idx]
        )

        J_1[idx] = J_i_duopoly(s_1, actions_1[idx], m[0])
        J_2[idx] = J_i_duopoly(s_2, actions_2[idx], m[1])
        J_1_deception[idx] = J_1_oblivious_duopoly(
            state_deception, delta_2, p, m[0], S_d
        )
        J_2_deception[idx] = J_i_duopoly(s_2_deception, actions_2_deception[idx], m[1])

        delta[idx] = delta_2

        reaction_curves_deception[idx] = RC_1_deceptive(x_deception, m, S_d, p, delta_2)

        if idx == len(time) - 1:
            continue

        nominal_gradient = np.array(
            [
                J_1_duopoly_grad_1(x_nominal, p, m[0], S_d),
                J_2_duopoly_grad_2(x_nominal, p, m[1]),
            ]
        )

        deceptive_gradient = np.array(
            [
                J_1_duopoly_grad_1(x_deception, p, m[0], S_d)
                + delta_2 * J_1_duopoly_grad_2(x_deception, p, m[0]),
                J_2_duopoly_grad_2(x_deception, p, m[1]),
            ]
        )


def plot_duo(simulation, J_2_ref):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "serif", "mathtext"})


def anima_reaction_curves(
    simulation,
    m,
    p,
    S_d,
    isoprofit_levels=None,
    x1_limits=(25.0, 70.0),
    frame_step=20,
    interval=60,
):
    delta_values = np.asarray(simulation.delta, dtype=float)
    actions_1_deception = simulation.actions_1_deception
    actions_2_deception = simulation.actions_2_deception
    time_values = simulation.time

    x_1 = np.linspace(x1_limits[0], x1_limits[1])
    rotation_point = np.array([m[0], m[0] - S_d * p])
    frame_indices = np.arange(0, len(delta_values), frame_step, dtype=int)
    sampled_delta = delta_values[frame_indices]
    sampled_actions_1 = actions_1_deception[frame_indices]
    sampled_actions_2 = actions_2_deception[frame_indices]
    sampled_time = time_values[frame_indices]
    rc_curves = np.array(
        [RC_1_deceptive_x2(x_1, m, S_d, p, delta_2) for delta_2 in sampled_delta]
    )

    isoprofit_levels = [] if isoprofit_levels == None else list(isoprofit_levels)

    isoprofit_curves = [
        (level, *isoprofit_2(x_1, m, p, level)) for level in isoprofit_levels
    ]

    return
