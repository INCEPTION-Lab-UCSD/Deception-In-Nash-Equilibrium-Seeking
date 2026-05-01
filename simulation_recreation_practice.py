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
