import numpy as np

import visualizations

x = np.array([50, 110 / 3])
a = 0.05
k = 0.03
omega_1 = 7877.75
omega_2 = 7436.5
epsilon = -0.001
J_2_ref = 1000
u = np.array([0, 0])
m = np.array([30, 30])
p = 0.2
S_d = 100
t = 100
visualizations.run_duopoly(x, a, k, omega_1, omega_2, J_2_ref, epsilon, S_d, u, p, m, t)
