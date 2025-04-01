from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import numpy as np
import tqdm
f01_1 = np.logspace(np.log10(0.1), np.log10(1), num=10)
f1_10 = np.logspace(np.log10(1), np.log10(10), num=10)
f10_100 = np.logspace(np.log10(10), np.log10(100), num=10)
f100_1000 = np.logspace(np.log10(100), np.log10(1000), num=10)
# f10_100 = np.arange(10, 101, 10)
f = f100_1000
t = np.arange(0, 1, 0.0001)
# best_M = float('inf')
# for i in tqdm.trange(500000, colour='green'):
#     phi = np.random.uniform(0, 2*np.pi, 10)
#     s = np.sum(np.sin(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0)
#     M = np.max(np.abs(s))
#     if M < best_M:
#         best_M = M
#         best_phi = phi
#         print(i, best_M, best_phi.tolist())

def s(amp_phase):
    # Extract the amplitude and phase shifts from the input array
    amp = amp_phase[0:10]
    phase_shifts = amp_phase[10:]
    # Generate the signal using the given phase shifts
    s = np.sum(amp[:, None]*np.sin(2 * np.pi * f[:, None] * t + phase_shifts[:, None]), axis=0)
    # Return the maximum absolute value of the signal
    return np.max(np.abs(s))
constraints = [
    {'type': 'ineq', 'fun': lambda x: s(x) - 0.2},
    {'type': 'ineq', 'fun': lambda x: -s(x) + 0.3}
]
bounds = Bounds(lb=[0.07]*10+[0]*10, ub=[1]*10 +[2*np.pi]*10)
opt_result = minimize(s, np.zeros(20), method='SLSQP', bounds=bounds,constraints=constraints, options={'maxiter': 5000000, 'disp': True})
print("Amplitudes:", opt_result.x[:10].tolist())
print("Phase shifts:", opt_result.x[10:].tolist())
print("Minimum max amplitude:", opt_result.fun)
# exit()
# Plot the signal with the optimal phase shifts

amp = opt_result.x[0:10]
phase_shifts = opt_result.x[10:]
y = 1+np.sum(amp[:, None]*np.sin(2 * np.pi * f[:, None] * t + phase_shifts[:, None]), axis=0)
plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.title("Signal with Optimal Phase Shifts")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
