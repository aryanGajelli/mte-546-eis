from read_bin import load_data
from data_cleaner import break_data, break_hf
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from constants import *


def fit_sine_wave(t, y, f=1):
    def sin_func(t, A, phi, offset):
        return A * np.sin(f * 2 * np.pi * t + phi) + offset
    # Initial guess for the parameters: A, w, phi, offset
    A_guess = (np.max(y) - np.min(y)) / 2
    # w_guess = 2 * np.pi  # Frequency estimate
    offset_guess = np.mean(y)  # Offset estimate

    initial_guess = [A_guess, 0, offset_guess]

    # Fit the sine wave to the data
    params, _ = optimize.curve_fit(sin_func, xdata=t, ydata=y, p0=initial_guess, method='trf', maxfev=100000,
                                   bounds=([-np.inf, -2*np.pi, -np.inf], [np.inf, 2 * np.pi, np.inf]))
    print(f"{params[0]:.6f}sin({f:.3f}*2*pi*t + {params[1]:.6f}) + {params[2]:.6f}")
    return sin_func(t, *params), params[0], params[1]


df, fs = load_data(cell=79, temp='35C', soc=50)
lf_v, hf_v = break_data(df, volt=True)
lf_i, hf_i = break_data(df, volt=False)

f1_10v, f10_100v, f100_1000v = break_hf(hf_v)

f1_10i, f10_100i, f100_1000i = break_hf(hf_i)

tv = f1_10v[0].index
ti = f1_10i[0].index
v, *v_params = fit_sine_wave(tv, f1_10v[0], f=1)
i, *i_params = fit_sine_wave(ti, f1_10i[0], f=1)

v_freq = v_params[0] * np.exp(1j * v_params[1])
i_freq = i_params[0] * np.exp(1j * i_params[1])


# for i, (v_param, i_param) in enumerate(zip(v_params, i_params)):
#     v_freq = v_param * np.exp(1j * (PHI[i] + v_offset))
#     i_freq = i_param * np.exp(1j * (PHI[i]))
impedance = (v_freq / -i_freq)
impedance = impedance.conjugate()
print(f"Impedance at {FREQUENCY_SWEEP_F1_1000[0]} Hz: {impedance:.6f}")

