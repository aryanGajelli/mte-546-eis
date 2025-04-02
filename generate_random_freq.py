import numpy as np
import random
import math

def generate_random_phase_shifts(num_shifts):
    # Generate random phase shifts using a normal distribution
    phase_shifts = []
    for _ in range(num_shifts):
        # Generate a random phase shift between 0 and 2pi
        phase_shift = random.uniform(0, 2 * math.pi)
        phase_shifts.append(phase_shift)
    return phase_shifts

def generate_freq_log_spacing(start_freq, end_freq, num_points):
    # Generate logarithmically spaced frequencies between start_freq and end_freq
    return np.logspace(np.log10(start_freq), np.log10(end_freq), num_points)

print("Random phase shifts:")
phase_shifts = generate_random_phase_shifts(10)
for i, shift in enumerate(phase_shifts):
    print(f"Shift {i+1}: {shift:.2f} radians")

freq_log_spacing = generate_freq_log_spacing(0.100, 1, 10)
print("\nLogarithmically spaced frequencies 0.1-1 Hz:")
print(freq_log_spacing.tolist())

#log spacing from 1Hz to 1000Hz while excluding 1Hz
freq_log_spacing = generate_freq_log_spacing(1, 10, 10)
print("\nLogarithmically spaced frequencies 1-10 Hz:")
print(freq_log_spacing.tolist())
# for i, freq in enumerate(freq_log_spacing):
#     print(f"Frequency {i+1}: {freq:.2f} Hz")

freq_log_spacing = generate_freq_log_spacing(10, 100, 10)
print("\nLogarithmically spaced frequencies 10-100 Hz:")
print(freq_log_spacing.tolist())

freq_log_spacing = generate_freq_log_spacing(100, 1000, 10)
print("\nLogarithmically spaced frequencies 100-1000 Hz:")
print(freq_log_spacing.tolist())

# shifts = [0.38, 0.67, 1.86, 4.71, 0.77, 5.42, 1.38, 0.24, 1.69, 5.36]
# log_freqs = [0.1, 0.13, 0.16, 0.21, 0.27, 0.34, 0.43, 0.55, 0.71, 0.9]

# for i in range(len(shifts)):
#     print(f"Frequency: {log_freqs[i]:.2f} Hz, Phase Shift: {shifts[i]:.2f} radians, Cycles: {50*log_freqs[i]:.2f} cycles")

# Frequency: 0.10 Hz, Phase Shift: 0.38 radians, Cycles: 5.00 cycles
# Frequency: 0.13 Hz, Phase Shift: 0.67 radians, Cycles: 6.50 cycles
# Frequency: 0.16 Hz, Phase Shift: 1.86 radians, Cycles: 8.00 cycles
# Frequency: 0.21 Hz, Phase Shift: 4.71 radians, Cycles: 10.50 cycles
# Frequency: 0.27 Hz, Phase Shift: 0.77 radians, Cycles: 13.50 cycles
# Frequency: 0.34 Hz, Phase Shift: 5.42 radians, Cycles: 17.00 cycles
# Frequency: 0.43 Hz, Phase Shift: 1.38 radians, Cycles: 21.50 cycles
# Frequency: 0.55 Hz, Phase Shift: 0.24 radians, Cycles: 27.50 cycles
# Frequency: 0.71 Hz, Phase Shift: 1.69 radians, Cycles: 35.50 cycles
# Frequency: 0.90 Hz, Phase Shift: 5.36 radians, Cycles: 45.00 cycles

