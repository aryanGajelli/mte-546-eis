
from constants import *
from pathlib import Path
import array
import struct
from collections import namedtuple
import numpy as np
from scipy import signal
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
TYPE_DIGITAL = 0
TYPE_ANALOG = 1
expected_version = 0

AnalogData = namedtuple('AnalogData', ('begin_time', 'sample_rate', 'downsample', 'num_samples', 'samples'))


def parse_analog(fpath: Path):
    with open(fpath, 'rb') as f:
        # Parse header
        identifier = f.read(8)
        if identifier != b"<SALEAE>":
            raise Exception("Not a saleae file")

        version, datatype = struct.unpack('=ii', f.read(8))

        if version != expected_version or datatype != TYPE_ANALOG:
            raise Exception("Unexpected data type: {}".format(datatype))

        # Parse analog-specific data
        begin_time, sample_rate, downsample, num_samples = struct.unpack('=dqqq', f.read(32))

        # Parse samples
        # samples = array.array("f")
        # samples.fromfile(f, num_samples)
        samples = np.fromfile(f, dtype=np.float32, count=num_samples, sep='')
        q = sample_rate//DOWNSAMPLE_RATE
        samples = signal.decimate(samples, q, ftype='fir')

    return AnalogData(begin_time, sample_rate/q, downsample, len(samples), samples)


def load_data(cell: int, temp: TEMP_RANGES, soc: SOC_RANGES, discharge_current: float = 1):
    # load three sets of data from the analog bin files and store in single dataframe
    data_dir_i = Path(f'data/cell{cell}/{temp}/{soc}/analog_5.bin')
    data_dir_temp = Path(f'data/cell{cell}/{temp}/{soc}/analog_4.bin')
    data_dir_v = Path(f'data/cell{cell}/{temp}/{soc}/analog_6.bin')
    start = time.time()
    data = [
        parse_analog(data_dir_i),
        parse_analog(data_dir_temp),
        parse_analog(data_dir_v)
    ]

    t = np.linspace(0, data[0].num_samples/data[0].sample_rate, data[0].num_samples, endpoint=False)

    # samples_current =
    offset = 0.23216784
    # offset = get_v_offset(samples_current, time, discharge_current)

    df = pd.DataFrame({
        'I': data[0].samples * V_2_A - offset,
        'T': (data[1].samples + TEMPERATURE_CHANNEL_OFFSET) / 3 * 340 - 70,
        'V': data[2].samples + VOLTAGE_CHANNEL_OFFSET
    }, index=t, columns=['I', 'T', 'V'])
    # print(df)
    return df, data[0].sample_rate


if __name__ == "__main__":
    df, _ = load_data(79, '25C', 100)

    plt.plot(df.index, df['V'])
    plt.xlabel('Time (s)')
    # plt.ylabel('Current (A)')
    # plt.title('Current over Time')
    plt.grid()
    # plt.legend()
    plt.show()
