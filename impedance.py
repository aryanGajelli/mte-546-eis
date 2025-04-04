import pandas as pd
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
    # print(f"{params[0]:.6f}sin({f:.3f}*2*pi*t + {params[1]:.6f}) + {params[2]:.6f}")
    return sin_func(t, *params), params[0], params[1]

def get_impedances(cell, temp, soc):
    df, fs = load_data(cell, temp, soc)
    lf_v, hf_v = break_data(df, volt=True)
    lf_i, hf_i = break_data(df, volt=False)

    f1_10v, f10_100v, f100_1000v = break_hf(hf_v)
    fv = f1_10v + f10_100v + f100_1000v

    f1_10i, f10_100i, f100_1000i = break_hf(hf_i)
    fi = f1_10i + f10_100i + f100_1000i
    impedances = []
    for i in range(len(fv)):
        f = FREQUENCY_SWEEP_F1_1000[i]
        tv = fv[i].index
        ti = fi[i].index
        _, *v_params = fit_sine_wave(tv, fv[i], f=f)
        _, *i_params = fit_sine_wave(ti, fi[i], f=f)

        v_freq = v_params[0] * np.exp(1j * v_params[1])
        i_freq = i_params[0] * np.exp(1j * i_params[1])

        impedance = (v_freq / -i_freq)
        impedance = impedance.conjugate()
        impedances.append(impedance)
    
    return pd.Series(impedances, index=FREQUENCY_SWEEP_F1_1000, name=f"Impedance {cell} {temp} {soc}% SOC")

z = get_impedances(79, '45C', 10)
plt.plot(z.values.real, z.values.imag, label='Cell 79 45C 100% SOC')
plt.xlabel('Real Part (Ohm)')
plt.ylabel('Imaginary Part (Ohm)')
plt.title('Nyquist Plot')
plt.grid()
plt.legend()
plt.show()



