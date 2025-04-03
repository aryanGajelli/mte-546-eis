from os import path
import shutil
import json
import argparse
import pathlib

# Usage:
# first locate the calibration file for your device. Calibration files are stored here: C:\Users\<your user folder>\AppData\Roaming\Logic\calibrations
# the first part of the file name is the base 10 version of the device ID.
# If there is more than 1 .cal file, you can copy the hex device ID shown in the device information dialog into a hex to decimal converter online, to locate the correct cal file.
# copy the absolute path to the calibration file, then run the following command:
# python saleae_cal_editor.py "c:\path\to\calibration\file.cal" --ch0-gain 1.001 --ch1-offset 0.12 --ch2-gain 5.002 --ch6-offset -1.3
# use arguments --ch0-gain and --ch0-offset through --ch15-gain and --ch15-offset to optionally edit each analog channel.
# the displayed voltage will first be multiplied by the gain, then subtracted by the offset.


# Plan:
# 1. take calibration file path as argument
# 2. check if backup exists (.cal.bak), and if not, create it.
# 3. load backup file
# 4. apply transforms
# 5. generate new calibration file, over-writing the original.


parser = argparse.ArgumentParser(
    description='Saleae Calibration Range Editor')

parser.add_argument('calibrationfile', type=pathlib.Path)
for i in range(0, 16):
    parser.add_argument(
        f'--ch{i}-gain', help=f'apply gain to channel {i}', type=float, default=1)
    parser.add_argument(f'--ch{i}-offset',
                        help=f'apply offset to channel {i}', type=float, default=0)

args = parser.parse_args()
print(args.calibrationfile)

# calibration_path = "C:/Users/markg/AppData/Roaming/Logic/calibrations/4824640135602372227-1550281708727-1.cal"
calibration_path = args.calibrationfile

if not path.exists(calibration_path):
    print(f'calibration file \"{calibration_path}\" does not exist')
    exit(1)
gains = [
    args.ch0_gain,
    args.ch1_gain,
    args.ch2_gain,
    args.ch3_gain,
    args.ch4_gain,
    args.ch5_gain,
    args.ch6_gain,
    args.ch7_gain,
    args.ch8_gain,
    args.ch9_gain,
    args.ch10_gain,
    args.ch11_gain,
    args.ch12_gain,
    args.ch13_gain,
    args.ch14_gain,
    args.ch15_gain,
]
offsets = [
    args.ch0_offset,
    args.ch1_offset,
    args.ch2_offset,
    args.ch3_offset,
    args.ch4_offset,
    args.ch5_offset,
    args.ch6_offset,
    args.ch7_offset,
    args.ch8_offset,
    args.ch9_offset,
    args.ch10_offset,
    args.ch11_offset,
    args.ch12_offset,
    args.ch13_offset,
    args.ch14_offset,
    args.ch15_offset,
]

print(gains)
print(offsets)


calibration_path_backup = f'{calibration_path}.bak'

if not path.exists(calibration_path_backup):
    calibration_path_backup = shutil.copyfile(
        calibration_path, calibration_path_backup)

print(f'target file: {calibration_path}')
print(f'backup path: {calibration_path_backup}')

source_file = open(calibration_path_backup)
calibration = json.load(source_file)
source_file.close()

original_voltage_ranges = calibration['data']['fullScaleVoltageRanges']

channel_count = len(original_voltage_ranges)

for i in range(0, channel_count):
    channel_range = original_voltage_ranges[i]
    channel_range['minimumVoltage'] = channel_range['minimumVoltage'] * \
        gains[i] - offsets[i]
    channel_range['maximumVoltage'] = channel_range['maximumVoltage'] * \
        gains[i] - offsets[i]


with open(calibration_path, 'w') as destination_file:
    json.dump(calibration, destination_file, ensure_ascii=False, indent=4)