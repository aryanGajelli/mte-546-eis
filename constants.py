from typing import Literal
import numpy as np

SOC_RANGES = Literal[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
TEMP_RANGES = Literal['25C', '35C']
A_2_V = 10/60
V_2_A = 60/10
VOLTAGE_CHANNEL_OFFSET = 15/1000
TEMPERATURE_CHANNEL_OFFSET = 19/1000

DOWNSAMPLE_RATE = 100000

FREQ_MULTI_SINE = np.geomspace(0.1, 1, 5)
PHI = np.array([1.354738, 4.254050, 2.726734, 4.975810, 1.100473])
AMP = np.array([0.106223, 0.103442, 0.104527, 0.113251, 0.105519])
F1_10 = np.geomspace(1, 10, 10)
F10_100 = np.geomspace(10, 100, 10)
F100_1000 = np.geomspace(100, 1000, 10)
FREQUENCY_SWEEP_F1_1000 = np.concatenate((F1_10, F10_100, F100_1000))