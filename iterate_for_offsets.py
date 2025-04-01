import numpy as np
f = np.array([0.1, 0.13, 0.16, 0.21, 0.27])
t = np.arange(0, 100, 0.01)
best_M = float('inf')
for _ in range(10000000):
    phi = np.random.uniform(0, 2*np.pi, 5)
    s = np.sum(np.sin(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0)
    M = np.max(np.abs(s))
    if M < best_M:
        best_M = M
        best_phi = phi

print(best_phi)