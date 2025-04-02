import numpy as np
N = 10
f = np.geomspace(0.1, 1, N)
t = np.arange(0, 100, 0.01)
best_M = float('inf')
for _ in range(10000):
    phi = np.random.uniform(0, 2*np.pi, N)
    s = np.sum(np.sin(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0)
    M = np.max(np.abs(s))
    if M < best_M:
        best_M = M
        best_phi = phi
        print(best_M, best_phi)