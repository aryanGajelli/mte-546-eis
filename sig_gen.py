import numpy as np
import math
import random

N_CYCLES = 4
A_2_V = 0.177
dc_offsset = 1 * A_2_V
amplitude = 0.3 * A_2_V

f01_1 = np.logspace(np.log10(0.1), np.log10(1), num=10)
f1_10 = np.logspace(np.log10(1), np.log10(10), num=10)
f10_100 = np.logspace(np.log10(10), np.log10(100), num=10)
f100_1000 = np.logspace(np.log10(100), np.log10(1000), num=10)

def o(a,b):
    return f"STEP([T]-{a:.3f})*STEP({b:.3f}-[T])"

out = f"{dc_offsset:0.3f} + {amplitude:0.3f}*"

IN_BETWEEN = 0.1

for i, f in enumerate(f1_10):

    out += o(i+IN_BETWEEN*i, i+N_CYCLES/f+IN_BETWEEN*i) + f"*SIN({'-' if i == 0 else ''}{f:.3f}*2*[PI]*([T]-{IN_BETWEEN*i:.3f}){'-[PI]' if i == 0 else ''})"
    if i != len(f1_10) - 1:
        out += " + "


print(out)
