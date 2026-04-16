import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# [1] Duopoly game
def J_i_duopoly(s_i, x_i, m_i):
    return -s_i * (x_i - m_i)


def J_1_duopoly_grad_1(x, p, m_1, S_d):
    return (2 * x[0] / p) - (m_1 / p) - (x[1] / p) - S_d


def J_1_duopoly_grad_2(x, p, m_1):
    return -(x[0] / p) + (m_1 / p)


def J_2_duopoly_grad_2(x, p, m_2):
    return -(x[0] / p) - (2 * x[1] / p) + (m_2 / p)


# deceived cost function for this problem
def J_1_oblivious_duopoly(x, delta_2, p, m_1, S_d):
    s_1_cur = s_1_duopoly(x[0], x[1], p, S_d)
    J_1 = -(s_1_cur + (delta_2 / (2 * p)) * (x[0] - m_1))(x[0] - m_1)
    return J_1


def s_1_duopoly(x_1, x_2, p, S_d):
    return S_d - s_2_duopoly(p, x_1, x_2)


def s_2_duopoly(p, x_1, x_2):
    return 1 / p * (x_1 - x_2)


def NE_duopoly_1(m_1, m_2, S_d, p):
    return 1 / 3 * (2 * m_1 + m_2 + 2 * S_d * p)


def NE_duopoly_2(m_1, m_2, S_d, p):
    return 1 / 3 * (m_1 + 2 * m_2 + S_d * p)


# generic NE dynamics for x_1
def x_1_duopoly(u_1, a, omega_1, t):
    return u_1 + a * np.sin(omega_1 * t)


# deception NE dynamics for x_2
def x_2_duopoly(u_2, a, omega_1, omega_2, delta_2, t):
    return u_2 + a * (np.sin(omega_2 * t) + delta_2 * np.sin(omega_1 * t))


def delta_2_update_duopoly_deception(x, epsilon, J_2_ref, p, m_2):
    s_i = s_2_duopoly(p, x[0], x[1])

    return epsilon * (J_i_duopoly(s_i, x[1], m_2) - J_2_ref)


def run_duopoly(x, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, u, p, m, t):
    x_deception = x
    actions_1 = [x[0]]
    actions_2 = [x[1]]
    actions_deception_1 = [x[0]]
    actions_deception_2 = [x[1]]

    delta = []

    u_deception = u
    J_1s = []
    J_2s = []
    J_1_deception = []
    J_2_deception = []
    for i in range(t):
        actions_1.append(x[0])
        actions_2.append(x[1])
        actions_deception_1.append(x_deception[0])
        actions_deception_2.append(x_deception[1])
        s_1 = s_1_duopoly(x[0], x[1], p, S_d)
        s_2 = s_2_duopoly(x[0], x[1], p)
        s_1_deception = s_1_duopoly(x_deception[0], x_deception[1], p, S_d)
        s_2_deception = s_2_duopoly(x_deception[0], x_deception[1], p)
        J_1_NE = J_i_duopoly(s_1, x[0], m[0])
        J_2_NE = J_i_duopoly(s_2, x[1], m[1])
        delta_2 = delta_2_update_duopoly_deception(x, epsilon, J_2_ref, p, m[1])
        J_1_DNE = J_1_oblivious_duopoly(x_deception, delta_2, p, m[0], S_d)
        J_2_DNE = J_i_duopoly(s_2_deception, x_deception, m[1])

        x[0] = u[0] + a * np.sin(omega_1 * t)
        x[1] = u[1] + a * np.sin(omega_2 * t)
        u_1_change = -k * J_1_duopoly_grad_1(u, p, m[0], S_d)
        u_2_change = -k * J_2_duopoly_grad_2(u, p, m[1])
        u_1_deception_change = -k * (
            J_1_duopoly_grad_1(u_deception, p, m[0], S_d)
            + delta_2 * J_1_duopoly_grad_2(u_deception, p, m[0])
        )
        u_2_deception_change = -k * J_2_duopoly_grad_2(u_deception, p, m[1])

        u[0] += u_1_change
        u[1] += u_2_change
        u_deception[0] += u_1_deception_change
        u_deception[1] += u_2_deception_change

        x_deception[0] = u_deception[0] + a * np.sin(omega_1 * t)
        x_deception[1] = u_deception[1] + a * (
            np.sin(omega_2 * t) + delta_2 * np.sin(omega_1 * t)
        )

        J_1s.append(J_1_NE)
        J_2s.append(J_2_NE)
        J_1_deception.append(J_1_DNE)
        J_2_deception.append(J_2_DNE)
        delta.append(delta_2)
    actions_1.append(x[0])
    actions_2.append(x[1])
    actions_deception_1.append(x_deception[0])
    actions_deception_2.append(x_deception[1])

    return (
        J_1s,
        J_2s,
        J_1_deception,
        J_2_deception,
        actions_1,
        actions_2,
        actions_deception_1,
        actions_deception_2,
        delta,
    )
