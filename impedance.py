import read_bin as rb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from pathlib import Path
sys.path.append('../')

from typing import Literal

SOC_RANGES = Literal['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0']
A_2_V = 10/60
V_2_A = 60/10

def get_v_offset(data_in_amperage, time, discharge_current):
    start_t = None
    ref_val = None
    offset_values = []
    for idx, t in enumerate(time):
        if start_t is None:
            start_t = t
            ref_val = data_in_amperage[idx]
            offset_values.append(data_in_amperage[idx])
        elif (ref_val - 0.05 < data_in_amperage[idx] < ref_val + 0.05) and (discharge_current - 0.4 < data_in_amperage[idx] < discharge_current + 0.4):
            offset_values.append(data_in_amperage[idx])
        else:
            if start_t is not None:
                if t - start_t > 0.98:
                    return -(discharge_current - np.mean(offset_values))
                start_t = None
                current_val = None
                offset_values = []
    return 0

def load_data(soc: SOC_RANGES, cell_number: int, discharge_current: float):
    # load three sets of data from the analog bin files and store in single dataframe
    data_dir = Path(f'data/cell{cell_number}/{soc}/analog_5.bin')
    print(data_dir)
    with open(data_dir, 'rb') as f:
        data = rb.parse_analog(f)
    data_in_amperage = V_2_A * np.array(data.samples)
    time = np.linspace(0, data.num_samples/data.sample_rate, data.num_samples, endpoint=False)
    df = pd.DataFrame(data_in_amperage-get_v_offset(data_in_amperage, time, discharge_current), index=time, columns=['Current'])
    return df

df = load_data('100', 79, 1)
plt.plot(df.index, df['Current'])
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Current vs Time")
plt.grid()
plt.show()