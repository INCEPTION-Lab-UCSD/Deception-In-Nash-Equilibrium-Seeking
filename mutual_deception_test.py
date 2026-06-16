import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ── Parameters ────────────────────────────────────────────────────────────────
h = 0.0001
S = 100
p = 0.2
m1 = 30
m2 = 30

# ── Matrices ──────────────────────────────────────────────────────────────────
Q1 = 2 * np.array([[-1 / p, 1 / (2 * p)], [1 / (2 * p), 0]])

b1 = np.array([(m1 + S * p) / p, -m1 / p])

Q2 = 2 * np.array([[0, 1 / (2 * p)], [1 / (2 * p), -1 / p]])

b2 = np.array([-m2 / p, m2 / p])

# A0 uses row 0 of Q1 and row 1 of Q2  (MATLAB rows 1 and 2)
A0 = np.array([Q1[0, :], Q2[1, :]])

b0 = np.array([b1[0], b2[1]])  # MATLAB b1(1)  # MATLAB b2(2)

x0 = -np.linalg.solve(A0, b0)

# ── Scalar constants ───────────────────────────────────────────────────────────
c1 = -(Q1[0, 0] + Q2[1, 1]) / Q1[0, 1]
c2 = np.linalg.det(A0) / (Q2[0, 1] * Q1[1, 1] - Q2[1, 1] * Q1[0, 1])

# ── Direction vector and projections ──────────────────────────────────────────
w = np.array([1, -Q2[0, 1] / Q2[1, 1]])

q1 = -(b1[1] + Q1[1, :] @ x0)
q2 = Q1[1, :] @ w
q3 = Q1[0, :] @ w
r2 = 0.5 * w @ Q2 @ w
r1 = (b2 + Q2 @ x0) @ w

# ── Excitation / algorithm hyperparameters ────────────────────────────────────
a = 0.05
k = -0.03
w1 = 11877.75
w2 = 12436.5
epsilon1 = -0.001
epsilon2 = -0.0005
J1ref = 1200.0
J2ref = 1800.0


# ── Objective functions ───────────────────────────────────────────────────────
def J1(x):
    """Player 1 payoff."""
    return 0.5 * x @ Q1 @ x + b1 @ x - 3000.0


def J2(x):
    """Player 2 payoff."""
    return 0.5 * x @ Q2 @ x + b2 @ x


def J2quad(eps):
    return r2 * eps**2 + r1 * eps + J2(x0)


# ── Price perturbation ────────────────────────────────────────────────────────
def prices(t, u, d1, d2):
    return u + a * np.array(
        [
            np.sin(w1 * t) + d1 * np.sin(w2 * t),
            np.sin(w2 * t) + d2 * np.sin(w1 * t),
        ]
    )


# ── State derivative (4-component) ────────────────────────────────────────────
def udot(time, u):
    """
    State: u = [u1, u2, delta1, delta2]
    Returns du/dt as a 4-vector.
    """
    x = prices(time, u[:2], u[2], u[3])
    J1v = J1(x)
    J2v = J2(x)
    return np.array(
        [
            (-2.0 * k / a) * J1v * np.sin(w1 * time),
            (-2.0 * k / a) * J2v * np.sin(w2 * time),
            epsilon1 * (J1v - J1ref),
            epsilon2 * (J2v - J2ref),
        ]
    )


# ── Initial condition ─────────────────────────────────────────────────────────
u0 = np.array([50.0, 100.0 / 3.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
# Option 1 – Fixed-step RK4 (kept for reference; may diverge for long horizons)
# ══════════════════════════════════════════════════════════════════════════════
def run_fixed_rk4(horizon=50.0, dt=h):
    t_grid = np.arange(0.0, horizon + dt, dt)
    N = len(t_grid)
    u_hist = np.zeros((N, 4))
    u_hist[0] = u0
    J_hist = np.zeros((N - 1, 2))

    for i in range(N - 1):
        ui = u_hist[i]
        ti = t_grid[i]

        k1 = udot(ti, ui)
        k2 = udot(ti + 0.5 * dt, ui + 0.5 * dt * k1)
        k3 = udot(ti + 0.5 * dt, ui + 0.5 * dt * k2)
        k4 = udot(ti + dt, ui + dt * k3)

        u_hist[i + 1] = ui + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        J_hist[i, 0] = J1(u_hist[i, :2])
        J_hist[i, 1] = J2(u_hist[i, :2])

    return t_grid, u_hist, J_hist


# ══════════════════════════════════════════════════════════════════════════════
# Option 2 – Adaptive RK45 via scipy.integrate.solve_ivp  ← recommended
# ══════════════════════════════════════════════════════════════════════════════
def run_adaptive_rk45(horizon=1000.0, rtol=1e-6, atol=1e-8):
    """
    Adaptive-step RK45 integration.  The solver automatically shrinks the
    step near the fast sin(w*t) oscillations and grows it where the state
    changes slowly, preventing the exponential blow-up seen with fixed h.
    """
    print(f"Running adaptive RK45 over t = [0, {horizon}] ...")
    sol = solve_ivp(
        fun=udot,
        t_span=(0.0, horizon),
        y0=u0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True,  # allows evaluation at arbitrary t later
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    print(f"  Done. {sol.t.size} adaptive steps taken.")
    return sol


# ── Run adaptive solver ───────────────────────────────────────────────────────
sol = run_adaptive_rk45(horizon=1000.0)

# Evaluate on a uniform grid for plotting (avoids storing 10^7 raw points)
t_plot = np.linspace(0.0, 1000.0, 200_000)
u_plot = sol.sol(t_plot)  # shape (4, len(t_plot))

J1_plot = np.array([J1(u_plot[:2, i]) for i in range(len(t_plot))])
J2_plot = np.array([J2(u_plot[:2, i]) for i in range(len(t_plot))])

# ── Final equilibrium point ───────────────────────────────────────────────────
# Evaluate state at t = 1000*h*333334 ≈ 33.33  (same relative point as MATLAB)
t_final = 1000.0 * h * 333334  # ≈ 33.33
u_final = sol.sol(t_final)
delta1_f = u_final[2]

A0_mod = A0 + delta1_f * np.vstack([Q1[1, :], [0.0, 0.0]])
b0_mod = b0 + delta1_f * np.array([b1[1], 0.0])
xd = -np.linalg.solve(A0_mod, b0_mod)

print(f"\nx0      = {x0}")
print(f"xd      = {xd}")
print(f"u(T)    = {sol.sol(1000.0)}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Mutual Deception Duopoly – Adaptive RK45 (solve_ivp)", fontsize=13)

axes[0, 0].plot(t_plot, J1_plot, linewidth=0.4)
axes[0, 0].axhline(J1ref, color="r", linestyle="--", label=f"J1ref = {J1ref}")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("J1")
axes[0, 0].set_title("Player 1 payoff")
axes[0, 0].legend()

axes[0, 1].plot(t_plot, J2_plot, linewidth=0.4)
axes[0, 1].axhline(J2ref, color="r", linestyle="--", label=f"J2ref = {J2ref}")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("J2")
axes[0, 1].set_title("Player 2 payoff")
axes[0, 1].legend()

axes[1, 0].plot(t_plot, u_plot[0], linewidth=0.4, label="u1")
axes[1, 0].plot(t_plot, u_plot[1], linewidth=0.4, label="u2")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel("price estimate")
axes[1, 0].set_title("Price averages u1, u2")
axes[1, 0].legend()

axes[1, 1].plot(t_plot, u_plot[2], linewidth=0.4, label="δ1")
axes[1, 1].plot(t_plot, u_plot[3], linewidth=0.4, label="δ2")
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel("deception parameter")
axes[1, 1].set_title("Deception parameters δ1, δ2")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("../duopoly_simulation.png", dpi=150)
print("Plot saved.")
plt.show()
