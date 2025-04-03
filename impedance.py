from read_bin import load_data
from data_cleaner import break_data, break_hf
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


def sin_func(t, A, w, phi, offset):
    return A * np.sin(w * t + phi) + offset

def fit_sine_wave(t, y, w_guess = 0):
    # Initial guess for the parameters: A, w, phi, offset
    A_guess = (np.max(y) - np.min(y)) / 2
    # w_guess = 2 * np.pi  # Frequency estimate
    offset_guess = np.mean(y)  # Offset estimate

    initial_guess = [A_guess, w_guess, 0, offset_guess]

    # Fit the sine wave to the data
    params, _ = optimize.curve_fit(sin_func, xdata=t, ydata=y, p0=initial_guess, method='trf', maxfev=100000, bounds=([-np.inf, 0, -2*np.pi, -np.inf], [np.inf, np.inf, 2 * np.pi, np.inf]))
    print(f"{params=}")
    return sin_func(t, *params)

df, fs = load_data(cell=79, temp='35C', soc=0)
lf_v, hf_v = break_data(df, volt=True)
lf_i, hf_i = break_data(df, volt=False)

f1_10v, f10_100v, f100_1000v = break_hf(hf_v)

f1_10i, f10_100i, f100_1000i = break_hf(hf_i)

tv = f1_10v[0].index
ti = f1_10i[0].index
v = fit_sine_wave(tv, f1_10v[0], w_guess=2*np.pi*1)
i = fit_sine_wave(ti, f1_10i[0], w_guess=2*np.pi*1)

plt.plot(f1_10v[0], label='V')
plt.plot(f1_10i[0], label='I')
plt.plot(tv, v, linestyle='--', label='V fit')
plt.plot(ti, i, linestyle='--', label='I fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original and Reconstructed Waveforms')
plt.grid()
plt.legend()
plt.show()

