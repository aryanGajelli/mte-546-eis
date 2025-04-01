import numpy as np
import math
import random

A_2_V = 0.177
dc_offsset = 1 * A_2_V
amplitude = 0.15 * A_2_V

f01_1 = np.logspace(np.log10(0.1), np.log10(1), num=10)
f1_10 = np.logspace(np.log10(1), np.log10(10), num=10)
f10_100 = np.logspace(np.log10(10), np.log10(100), num=10)
f100_1000 = np.logspace(np.log10(100), np.log10(1000), num=10)

out = f"{dc_offsset:0.3f} + "

# shifts = [0.38, 0.67, 1.86, 4.71, 0.77, 5.42, 1.38, 0.24, 1.69, 5.36]
# shifts = [3.38435583, 6.26287887, 0.11899036, 1.49996256, 1.26403229, 0, 0, 0, 0, 0, 0]
amps = [0.07000021249117733, 0.07000003282155731, 0.07179721135046835, 0.07000011642146807, 0.07000005771742206, 0.0700018387420161, 0.07221661073722087, 0.07000007012767749, 0.07000104269444767, 0.07000003235210416]
shifts = [0.12478582466180406, 0.2515009650793344, 0.07550939329595774, 0.03359542692303297, 0.45310078571234824, 0.49144075780997015, 0.20416506698777742, 0.2303806790873049, 0.47861957763147855, 0.5269865270471885]
log_freqs = f10_100

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

NUM_SINES = 10

for i, (amp, freq, phase) in enumerate(zip(amps, log_freqs, shifts)):
    out += f"{amp * A_2_V:0.5f}*SIN({freq:0.3f}*2*[PI]*[T] + {phase:0.5f})"
    if i < NUM_SINES - 1:
        out += " + "
    else:
        break

print(out)
