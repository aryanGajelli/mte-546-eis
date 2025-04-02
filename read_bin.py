import array
import struct
import sys
from collections import namedtuple
import numpy as np

import matplotlib.pyplot as plt
TYPE_DIGITAL = 0
TYPE_ANALOG = 1
expected_version = 0

AnalogData = namedtuple('AnalogData', ('begin_time', 'sample_rate', 'downsample', 'num_samples', 'samples'))

def parse_analog(f):
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
    samples = array.array("f")
    samples.fromfile(f, num_samples)

    return AnalogData(begin_time, sample_rate, downsample, num_samples, samples)


if __name__ == '__main__':
    filename = sys.argv[1]
    print("Opening " + filename)

    with open(filename, 'rb') as f:
        data = parse_analog(f)

    # Print out all analog data
    print("Begin time: {}".format(data.begin_time))
    print("Sample rate: {}".format(data.sample_rate))
    print("Downsample: {}".format(data.downsample))
    print("Number of samples: {}".format(data.num_samples))

    print("  {0:>20} {1:>10}".format("Time", "Voltage"))
    A_2_V = 10/60
    V_2_A = 60/10
    OFFSET_A = 0.225
    t = np.linspace(0, data.num_samples/data.sample_rate, data.num_samples)
    a = V_2_A*np.array(data.samples) - OFFSET_A
    plt.plot(t, a)
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Raw Volt (V)")
    plt.show()