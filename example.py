import numpy as np
import matplotlib.pyplot as plt

# Time-domain signal
fs = 1000  # Sampling frequency (Hz)
Δt = 1 / fs  # Sampling period (s)
N = 1024  # Number of samples
t = np.arange(N) * Δt  # Time vector (s)
I_time = 2 * np.sin(2 * np.pi * 50 * t)  # Current: 2A amplitude, 50 Hz

# Compute FFT
I_freq = np.fft.fft(I_time)
f = np.fft.fftfreq(N, Δt)  # Frequency bins
I_mag = np.abs(I_freq) / N  # Normalize by N
I_mag = 2 * I_mag  # Double for single-sided spectrum (except DC)
I_mag[0] = I_mag[0] / 2  # DC component is not doubled

# Single-sided spectrum (positive frequencies)
f_pos = f[:N//2]
I_mag_pos = I_mag[:N//2]

# Plot
plt.figure(figsize=(12, 5))

# Time domain
plt.subplot(1, 2, 1)
plt.plot(t, I_time)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Time Domain')

# Frequency domain
plt.subplot(1, 2, 2)
plt.plot(f_pos, I_mag_pos)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Current Amplitude (A)')
plt.title('Frequency Domain (FFT)')
plt.tight_layout()
plt.show()