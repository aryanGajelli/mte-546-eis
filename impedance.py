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


def lpf(data, sample_rate, cuttoff_freq):
    order = 2
    fc = cuttoff_freq/(sample_rate/2)  # normalized
    b, a = signal.butter(order, fc, btype='low', analog=False)
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


def exp_decay(x, A, s, t, y0):
    return A*np.exp((x-s)*t) + y0


df, fs = load_data(cell=79, temp='25C', soc=100)

start_time = 9.32688 + 0.6
end_time = 39.3194

gradient = np.gradient(df['I'], 1/fs)
gradient_mask = np.abs(gradient) > 40000

# get the time of the first peak in the gradient
first_peak_time = df.index[gradient_mask][0]
start_time = first_peak_time 
end_time = start_time + 29.8

print("Start time:", start_time)
print("End time:", end_time)

extracted_voltage = df['V'][(df.index >= start_time) & (df.index <= end_time)]
extracted_current = df['I'][(df.index >= start_time) & (df.index <= end_time)]
# remove the DC offset
t = extracted_voltage.index - start_time


# params, cov = optimize.curve_fit(exp_decay, t, extracted_voltage, p0=[1, 0.02, -1/30, 1], method='lm', maxfev=100000)
# trend = exp_decay(t, *params)
# coeffs = np.polyfit(t, extracted_voltage, 5)
# trend = np.polyval(coeffs, t)
# spl = make_splrep(t, extracted_voltage, k=3, s=3000)
v_trend = lpf(extracted_voltage, fs, 0.04)
a_trend = lpf(extracted_current, fs, 0.04)
ac_voltage = extracted_voltage - v_trend
ac_current = extracted_current - a_trend

v_params = compute_sine_wave_parameters(ac_voltage, t, v_trend)
i_params = compute_sine_wave_parameters(ac_current, t, a_trend)

print(f"Voltage params: {v_params}")
print(f"Current params: {i_params}")

#remove last item from v_params and i_params
v_offset = v_params[-1]
i_offset = i_params[-1]
v_params = v_params[:-1]
i_params = i_params[:-1]

print(f"Voltage offset: {v_offset}")
print(f"Current offset: {i_offset}")

for i, (v_param, i_param) in enumerate(zip(v_params, i_params)):
    v_freq = v_param * np.exp(1j * (PHI[i] + v_offset))
    i_freq = i_param * np.exp(1j * (PHI[i]))
    impedance = (v_freq / -i_freq)
    impedance = impedance.conjugate()
    print(f"Impedance at {FREQ_MULTI_SINE[i]} Hz: {impedance:.6f}")



# plt.plot(t, i)
# plt.plot(t, v)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.title('Voltage over Time')
# plt.grid()
# plt.legend()
# plt.show()

# t = 9.3265 start
# t = 39.3239 end
# take the voltage and current data between these two points


# compute_sine_wave_parameters(ac_voltage, extracted_voltage.index)

# # fft_current = fft.fft(extracted_current)
# # freqs = fft.fftfreq(len(extracted_current), 1/5000)
# # plt.plot(freqs, np.abs(fft_current))

# n = len(extracted_voltage)

# # # fft
# fft_current = fft.fft(extracted_current)[:n]
# fft_voltage = fft.fft(ac_voltage)[:n]
# freqs = fft.fftfreq(len(extracted_current), 1/fs)[:n]

# mag_current = np.abs(fft_current)
# mag_current[1:] = mag_current[1:] * 2 / n
# mag_current[0] = mag_current[0] / n

# mag_voltage = np.abs(fft_voltage)
# mag_voltage[1:] = mag_voltage[1:] * 2 / n
# mag_voltage[0] = mag_voltage[0] / n

# # plt.plot(freqs, fft_voltage)
# # plt.xlim(xmin=0, xmax=1.4)
# # plt.ylim(ymin=0, ymax=1)
# # plt.xlabel("Frequency (Hz)")
# # plt.ylabel("Magnitude")
# # plt.title("FFT of Current")
# # plt.grid()
# # plt.show()


# # plt.plot(extracted_voltage.index, ac_voltage)
# # plt.show()
