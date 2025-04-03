from constants import *
from typing import Literal
from read_bin import load_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, optimize
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy.interpolate import make_splrep
import sys
from pathlib import Path
sys.path.append('../')


FREQ_MULTI_SINE = np.geomspace(0.1, 1, 5)
PHI = np.array([1.354738, 4.254050, 2.726734, 4.975810, 1.100473])
AMP = np.array([0.106223, 0.103442, 0.104527, 0.113251, 0.105519])
F1_10 = np.geomspace(1, 10, 10)
F10_100 = np.geomspace(10, 100, 10)
F100_1000 = np.geomspace(100, 1000, 10)
FREQUENCY_SWEEP_F1_1000 = np.concatenate((F1_10, F10_100, F100_1000))


def lpf(data, sample_rate, cuttoff_freq, fc=None, btype='lowpass'):
    order = 2
    if fc == None:
        fc = cuttoff_freq/(sample_rate/2)  # normalized
    b, a = signal.butter(order, fc, btype='lowpass', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def get_v_offset(data_in_amperage, time, discharge_current):
    start_t = None
    ref_val = None
    offset_values = []
    for idx, t in enumerate(time):
        if start_t is None:
            start_t = t
            ref_val = data_in_amperage[idx]
            offset_values.append(data_in_amperage[idx])
        elif (ref_val - 0.05 < data_in_amperage[idx] < ref_val + 0.05) and (discharge_current - 0.35 < data_in_amperage[idx] < discharge_current + 0.35):
            offset_values.append(data_in_amperage[idx])
        else:
            if start_t is not None:
                if t - start_t > 0.98 and t - start_t < 1.1:
                    print(f"Start time: {start_t}, End time: {t}, Duration: {t - start_t:.2f} seconds")
                    print(-(discharge_current - np.mean(offset_values)))
                    return -(discharge_current - np.mean(offset_values))
                start_t = None
                offset_values = []
    return 0


def multisine(t, a1, a2, a3, a4, a5, p):
    """
    Model for sum of 5 sine waves, each with its own constant offset.
    """
    y = 0
    amp = [a1, a2, a3, a4, a5]
    # phase = [p1, p2, p3, p4, p5]
    for i in range(5):
        phi_k = PHI[i]
        f_k = FREQ_MULTI_SINE[i]
        y += amp[i] * np.sin(2 * np.pi * f_k * t + phi_k + p)
    # return np.sum(a_k[:, None]*np.sin(2 * np.pi * FREQ_MULTI_SINE[:, None] * t + PHI[:, None]), axis=0)
    return y


def compute_initial_guesses(data, t):
    p0 = [0]*6
    return p0

    N = len(t)
    dt = t[1] - t[0]
    freqs = fftfreq(N, dt)
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data) * 2 / N  # Single-sided magnitude

    p0 = []
    for f in FREQ_MULTI_SINE:
        idx = np.argmin(np.abs(freqs - f))
        p0.append(fft_magnitude[idx])  # Amplitude estimate

    p0 = [0]*5 + PHI
    return p0


def compute_sine_wave_parameters(data, t, trend):
    p0 = compute_initial_guesses(data, t)
    popt, pcov = optimize.curve_fit(f=multisine, xdata=t, ydata=data, p0=p0, method='trf', maxfev=100000)
    # Extract parameters from popt
    for i in range(5):
        a_k = popt[i]
        phi_k = PHI[i] + popt[-1]  # Use known phase
        print(f"Sine wave {i+1}:")
        print(f"  a{i+1} = {a_k:.6f}")
        print(f"  phi{i+1} = {phi_k:.6f} radians (fixed)")
    # print(f'  time offset = {popt[-1]}')
    print("\nCovariance matrix:")
    print(pcov)

    reconstructed_waveform = multisine(t, *popt)

    # reconstructed_waveform = spl(t)
    # reconstructed_waveform = multisine(t, *(np.array([0.106223, 0.103442, 0.104527, 0.113251, 0.105519])*1.07))
    # Plot the original and reconstructed waveforms
    # return reconstructed_waveform
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, data + trend, label='Original Data', color='blue')
    # plt.plot(t, reconstructed_waveform + trend, label='Reconstructed Waveform', color='orange')
    # plt.plot(t, trend, label='DC trend', color='green')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Original and Reconstructed Waveforms')
    # # plt.legend()
    # plt.grid()
    # plt.show()

    return popt


def break_data(df, volt=True):
    """
    volt = False implies current data
    """
    V_PEAK_THRESHOLD = 0.01
    I_PEAK_THRESHOLD = 0.1
    threshold = V_PEAK_THRESHOLD if volt else I_PEAK_THRESHOLD

    data = df['V' if volt else 'I']
    gradient = np.gradient(data)
    gradient = np.abs(gradient/np.max(gradient))
    indexes = df.index[(gradient > threshold)]
    splits = np.split(data, indexes)
    count = 0
    lf = None
    hf = None
    for split in splits:
        if split.size > 1000:
            if count == 1:
                lf = data[(df.index > split.index[0] + 0.0001) & (df.index < split.index[-1] - 0.0001)]
            if count == 3:
                hf = data[(df.index > split.index[0] + 0.0001) & (df.index < split.index[-1])]
            count += 1

    assert lf is not None
    assert hf is not None

    lt = lf.index
    ht = hf.index

    return pd.Series(lpf(lf, 1/(lt[1]-lt[0]), 10), lt), pd.Series(lpf(hf, 1/(ht[1]-ht[0]), 4000), ht)


def break_hf(data):
    INIT_OFFSET = 1 - 0.00025
    MIDDLE_OFFSET = 0.1
    t = data.index
    t0 = t[0]

    lmask = t - t0 < 16
    lmask1 = t - t0 <= 20
    l = lpf(data, 1/(t[1]-t[0]), 0.1)[lmask]
    l10 = lpf(data, 1/(t[1]-t[0]), 0.5)[~lmask & lmask1]
    l100 = lpf(data, 1/(t[1]-t[0]), 2)[~lmask1]

    data = np.hstack((data[lmask] - l, data[~lmask & lmask1] - l10, data[~lmask1] - l100))

    f1_10 = []
    start = INIT_OFFSET
    for f in np.geomspace(1, 10, 10):
        end = start + 4/f
        f1_10.append(pd.Series(data[(t - t0 >= start) & (t - t0 <= end)], index=t[(t - t0 >= start) & (t - t0 <= end)]))
        start = end + MIDDLE_OFFSET

    f10_100 = []
    MIDDLE_OFFSET -= 0.0001
    start -= 0.00035
    for f in np.geomspace(10, 100, 10):
        end = start + 4/np.round(f, 6) + 0.00015
        if f == 10:
            end += 0.00033
        f10_100.append(pd.Series(data[(t - t0 >= start) & (t - t0 <= end)], index=t[(t - t0 >= start) & (t - t0 <= end)]))
        start = end + MIDDLE_OFFSET - 0.00008

    f100_1000 = []
    for f in np.geomspace(100, 1000, 10):
        end = start + 4/np.round(f, 6) + 0.00015
        f100_1000.append(pd.Series(data[(t - t0 >= start) & (t - t0 <= end)], index=t[(t - t0 >= start) & (t - t0 <= end)]))
        start = end + MIDDLE_OFFSET - 0.00005

    return f1_10, f10_100, f100_1000
