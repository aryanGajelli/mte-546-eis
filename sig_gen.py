A_2_V = 0.177
dc_offsset = 1 * A_2_V
amplitude = 0.3 * A_2_V

out = f"{dc_offsset:0.3f} + {amplitude:0.3f} * ("

freq_offset = [(0.1, 0.77), (0.1, 0.77)]

for i, (freq, phase) in enumerate(freq_offset):
    out += f"SIN({freq:0.3f}*2*[PI]*[T] + {phase:0.5f})"
    if i < len(freq_offset) - 1:
        out += " + "
out += ")"
print(out)
