import numpy as np
import math
import random

A_2_V = 0.177
dc_offset = 1 * A_2_V
amplitude = 0.3 * A_2_V

# out = f"{dc_offset:0.3f}+{amplitude:0.3f}*("

# # # shifts = [0.38, 0.67, 1.86, 4.71, 0.77, 5.42, 1.38, 0.24, 1.69, 5.36]
# # # shifts = [3.38435583, 6.26287887, 0.11899036, 1.49996256, 1.26403229, 0, 0, 0, 0, 0, 0]
# # # shifts_01_1Hz = [5.19224017, 1.54357512, 0.7936779,  1.38204157, 0.77358612, 6.08901331, 2.38185817, 4.65999505, 4.33829204, 0.51981305]
# # # log_freqs_01_1Hz = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.60, 0.77, 1]
# # shifts_01_1Hz = [0.962794, 5.511953, 5.162770, 3.657122, 1.202294, 1.117731, 5.134582, 2.986176, 0.977367, 3.166233]
# shifts_01_1Hz = [3.83434054, 3.14174222, 4.28644511, 5.43187019, 1.96275115, 0, 0, 0, 0, 0]
# log_freqs_01_1Hz = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.60, 0.77, 1]

NUM_SINES = 30

# # have frequencies from 1Hz to 1000Hz after one other multiplied by window function 
# for i, (freq, phase) in enumerate(zip(log_freqs_01_1Hz, shifts_01_1Hz)):
#     out += f"SIN({freq:0.3f}*2*[PI]*[T]+{phase:0.3f})"
#     if i < NUM_SINES - 1:
#         out += " + "
#     else:
#         break
# out += ")"
# print(out)

out = f"{dc_offset:0.4f}+("
freq = np.geomspace(0.1, 1, 5)
amp = [0.106223, 0.103442, 0.104527, 0.113251, 0.105519]
phi = [1.354738, 4.254050, 2.726734, 4.975810, 1.100473]
print(freq)
# cutout first 5 frequencies
# freq = freq[5:]
for i, (a, p, freq) in enumerate(zip(amp, phi, freq)):
    out+= (f"{a*A_2_V:0.6f}*SIN({freq:0.6f}*2*[PI]*[T]+{p:0.6f})")
    if i < 5 - 1:
        out += " + "    
    else:
        break
out += ")*(STEP([T])-STEP([T]-30))"

# f1_10 = np.geomspace(1, 10, 10)
# f10_100 = np.geomspace(10, 100, 10)
# f100_1000 = np.geomspace(100, 1000, 10)

# f = np.concatenate((f1_10, f10_100, f100_1000))
# print(f)

# # have frequencies from 1Hz to 1000Hz after one other multiplied by window function 
# cumulative_offset = 31
# for i, (freq, time_per_wave) in enumerate(zip(f, np.round(4/f, 8))):
#     offset = 0 
#     if i < 10: 
#         offset = 0.1
#     else:
#         offset = 0.1
    
#     out += f"{amplitude:0.3f}*SIN({freq:0.3f}*2*[PI]*([T]-{cumulative_offset:0.6f}))*(STEP([T]-{cumulative_offset:0.6f})-STEP([T]-{cumulative_offset+time_per_wave:0.6f}))"
#     if i < NUM_SINES - 1:
#         out += " + "
#     else:
#         break
#     cumulative_offset += time_per_wave + offset
# out += ")"

print(out)