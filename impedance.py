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
VOLTAGE_CHANNEL_OFFSET = 15/1000
TEMPERATURE_CHANNEL_OFFSET = 19/1000

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
    offset = get_v_offset(samples_current, time, discharge_current)
    data_amperage = samples_current - offset

    data_temperature = ((np.array(data[1].samples)/3)*340 - 70) + TEMPERATURE_CHANNEL_OFFSET
    data_voltage = np.array(data[2].samples) + VOLTAGE_CHANNEL_OFFSET
    
    df = pd.DataFrame({
        'Current': data_amperage,
        'Temperature': data_temperature,
        'Voltage': data_voltage
    }, index=time)
    return df

df = load_data('100', 79, 1)
plt.plot(df.index, df['Current'])
plt.xlabel("Time (s)")
plt.ylabel("Current")
plt.title("Current vs Time")
plt.grid()
plt.show()