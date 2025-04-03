import read_bin as rb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, optimize
from scipy.fft import fft, fftfreq
import sys 
from pathlib import Path
sys.path.append('../')

from typing import Literal

SOC_RANGES = Literal['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0']
A_2_V = 10/60
V_2_A = 60/10
VOLTAGE_CHANNEL_OFFSET = 15/1000
TEMPERATURE_CHANNEL_OFFSET = 19/1000

FREQ_MULTI_SINE = np.geomspace(0.1, 1, 5)
PHI = [1.354738, 4.254050, 2.726734, 4.975810, 1.100473]

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

def lpf(data, sample_rate):
    order = 2 
    fc = 1000/(sample_rate/2) # normalized
    b, a = signal.butter(order, fc, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def load_data(soc: SOC_RANGES, cell_number: int, discharge_current: float):
    # load three sets of data from the analog bin files and store in single dataframe
    data_dir_i = Path(f'data/cell{cell_number}/{soc}/analog_5.bin')
    data_dir_temp = Path(f'data/cell{cell_number}/{soc}/analog_4.bin')
    data_dir_v = Path(f'data/cell{cell_number}/{soc}/analog_6.bin')
   
    data = []
    for data_dir in [data_dir_i, data_dir_temp, data_dir_v]:
        with open(data_dir, 'rb') as f:
            data.append(rb.parse_analog(f))
    
    time = np.linspace(0, data[0].num_samples/data[0].sample_rate, data[0].num_samples, endpoint=False)
    samples_current = np.array(data[0].samples) * V_2_A
    offset = 0.23216784
    # offset = get_v_offset(samples_current, time, discharge_current)
    data_amperage = samples_current - offset

    data_temperature = ((np.array(data[1].samples)/3)*340 - 70) + TEMPERATURE_CHANNEL_OFFSET
    data_voltage = np.array(data[2].samples) + VOLTAGE_CHANNEL_OFFSET
    
    df = pd.DataFrame({
        'Current': data_amperage,
        'Temperature': lpf(data_temperature, sample_rate=data[1].sample_rate),
        'Voltage': lpf(data_voltage, sample_rate=data[2].sample_rate)
    }, index=time, columns=['Current', 'Temperature', 'Voltage'])

    return df, data[0].sample_rate

def multisine(t, a1, a2, a3, a4, a5):
    """
    Model for sum of 5 sine waves, each with its own constant offset.
    """
    y = 0
    for i in range(5):
        a_k = [a1, a2, a3, a4, a5][i]
        phi_k = round(PHI[i], 6)
        f_k = round(FREQ_MULTI_SINE[i], 6)
        y += a_k * np.sin(2 * np.pi * f_k * t + phi_k)
    return y

def compute_initial_guesses(data, t):
    N = len(t)
    dt = t[1] - t[0]
    freqs = fftfreq(N, dt)
    fft_data = fft(data)
    fft_magnitude = np.abs(fft_data) * 2 / N  # Single-sided magnitude
    
    p0 = []
    for f in FREQ_MULTI_SINE:
        idx = np.argmin(np.abs(freqs - f))
        p0.append(fft_magnitude[idx])  # Amplitude estimate
    return p0


def compute_sine_wave_parameters(data_amperage, t):
    p0 = compute_initial_guesses(data_amperage, t)
    popt, pcov = optimize.curve_fit(f=multisine, xdata=t, ydata=data_amperage, p0=p0, method='trf', maxfev=100000)
   # Extract parameters from popt
    for i in range(5):
        a_k = popt[i]
        phi_k = PHI[i]  # Use known phase
        print(f"Sine wave {i+1}:")
        print(f"  a{i+1} = {a_k:.6f}")
        print(f"  phi{i+1} = {phi_k:.6f} radians (fixed)")
    print("\nCovariance matrix:")
    print(pcov)

    reconstructed_waveform = multisine(t, *popt)
    # Plot the original and reconstructed waveforms
    plt.figure(figsize=(10, 6))
    plt.plot(t, data_amperage, label='Original Data', color='blue')
    plt.plot(t, reconstructed_waveform, label='Reconstructed Waveform', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original and Reconstructed Waveforms')
    plt.legend()
    plt.grid()
    plt.show()

df, fs = load_data('90', 79, 1)
plt.plot(df.index, df['Voltage'], label='Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage over Time')
plt.legend()
plt.show()

# t = 9.3265 start 
# t = 39.3239 end 
# take the voltage and current data between these two points
# start_time = 9.32688
# end_time = 39.3194
# extracted_voltage = df['Voltage'][(df.index >= start_time) & (df.index <= end_time)]
# extracted_current = df['Current'][(df.index >= start_time) & (df.index <= end_time)]

# #remove the DC offset 
# coeffs = np.polyfit(extracted_voltage.index, extracted_voltage, 1)
# trend = np.polyval(coeffs, extracted_voltage.index)
# ac_voltage = extracted_voltage - trend

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
