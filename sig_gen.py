import numpy as np
import math
import random

A_2_V = 0.177
dc_offsset = 1 * A_2_V
amplitude = 0.15 * A_2_V

out = f"{dc_offsset:0.3f} + {amplitude:0.3f} * ("

# shifts = [0.38, 0.67, 1.86, 4.71, 0.77, 5.42, 1.38, 0.24, 1.69, 5.36]
shifts = [3.38435583, 6.26287887, 0.11899036, 1.49996256, 1.26403229, 0, 0, 0, 0, 0, 0]
# shifts = [0, np.pi/1.3, np.pi/1.6, np.pi/2, 0, 5.42, 1.38, 0.24, 1.69, 5.36]
log_freqs = [0.1, 0.13, 0.16, 0.21, 0.27, 0.34, 0.43, 0.55, 0.71, 0.9]

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

NUM_SINES = 5

for i, (freq, phase) in enumerate(zip(log_freqs, shifts)):
    out += f"SIN({freq:0.3f}*2*[PI]*[T] + {phase:0.5f})"
    if i < NUM_SINES - 1:
        out += " + "
    else:
        break
out += ")"
print(out)
