from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
from sympy.physics.vector.printing import vpprint, vlatex
from sympy import lambdify
import tqdm
# sp.init_printing()
init_vprinting()
def printS(expr, label = None, **kwargs):
    if label:
        print(label)
    print(vpprint(expr, wrap_line=False, **kwargs))
    print()


f01_1 = np.geomspace(0.1, 1, 10)
f1_10 = np.geomspace(1, 10, 10)
f10_100 = np.geomspace(10, 100, 10)
f100_1000 = np.geomspace(100, 1000, 10)

print(np.sum(np.round(3/f1_10,1))+0.01*30 + 0.1*10+0.1*10)
exit()
f = f01_1
t = np.arange(0, 30, 0.01)
# best_M = float('inf')
# for i in tqdm.trange(500000, colour='green'):
#     phi = np.random.uniform(0, 2*np.pi, 10)
#     s = np.sum(np.sin(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0)
#     M = np.max(np.abs(s))
#     if M < best_M:
#         best_M = M
#         best_phi = phi
#         print(i, best_M, best_phi.tolist()
t = sp.symbols('t')
def sp_s(amp_phase):
    # sympy version of s
    amp = amp_phase[0:10]
    phase_shifts = amp_phase[10:]
    
    s = sp.Add(*[amp[i] * sp.sin(2 * np.pi * f[i] * t + phase_shifts[i]) for i in range(10)])
    return s

amp_phase = sp.symbols('a0:10 b0:10')
s = sp_s(amp_phase)
ds = sp.diff(s, t)
dds = sp.diff(s, t, 2)

ks = dds/(sp.sqrt(1+ds**2)**3)
g_func = lambdify(amp_phase, ks, modules='numpy')

# printS(s, 's')
# printS(ds, 'ds')
# printS(dds, 'dds')
printS(ks, 'ks')

exit()

def get_s(amp_phase):
    # Extract the amplitude and phase shifts from the input array
    amp = amp_phase[0:10]
    phase_shifts = amp_phase[10:]
    # Generate the signal using the given phase shifts
    s = np.sum(amp[:, None]*np.sin(2 * np.pi * f[:, None] * t + phase_shifts[:, None]), axis=0)
    return s



def max_s(amp_phase):
    return np.max(np.abs(get_s(amp_phase)))

def s(amp_phase):
    # Extract the amplitude and phase shifts from the input array
    return np.max(np.abs(np.gradient(np.gradient(get_s(amp_phase)))))

constraints = [
    {'type': 'ineq', 'fun': lambda x: max_s(x) - 0.3},
    {'type': 'ineq', 'fun': lambda x: -max_s(x) + 0.3}
]
bounds = Bounds(lb=[0.1]*10+[0]*10, ub=[0.3]*10 +[2*np.pi]*10)
opt_result = minimize(s, np.ones(20), method='COBYLA', bounds=bounds, constraints=constraints, options={'maxiter': 50000, 'disp': True})
print("Amplitudes:", opt_result.x[:10].tolist())
print("Phase shifts:", opt_result.x[10:].tolist())
print("Minimum max amplitude:", max_s(opt_result.x))
# exit()
# Plot the signal with the optimal phase shifts

amp = opt_result.x[0:10]
phase_shifts = opt_result.x[10:]
y = 1+get_s(opt_result.x)
plt.figure(figsize=(10, 6))
plt.axhline(y=1, color='orange', linestyle='-')
plt.axhline(y=1.3, color='g', linestyle='--')
plt.axhline(y=0.7, color='g', linestyle='--')
plt.plot(t, y)
plt.title("Signal with Optimal Phase Shifts")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
