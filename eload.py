import usbtmc
from usb.core import USBError
import numpy as np
import matplotlib.pyplot as plt
import time
from icecream import ic
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)
# Connect to the Keithley 2380-120-60


class Keithley2380:

    def __init__(self, vid: int, pid: int):
        self.vid = vid
        self.pid = pid
        connection_string = f"USB::0x{vid:04x}::0x{pid:04x}::INSTR"

        self.device = usbtmc.Instrument(connection_string)

        self.device.write("*CLS")  # Reset the device
        self.device.write("SYST:REM")  # Set to remote mode
        print(f"{Fore.GREEN}Connected to device: {connection_string}{Fore.RESET}")

    def load_enable(self) -> None:
        """
        Enable or disable the output of the electronic load.
        :param enable: True to enable output, False to disable.
        """
        self.device.write("INP ON")

    def load_disable(self) -> None:
        """
        Enable or disable the output of the electronic load.
        :param enable: True to enable output, False to disable.
        """
        self.device.write("INP OFF")

    def set_current(self, current: float) -> None:
        """
        Set the current of the electronic load.
        :param current: Current in Amperes.
        """
        self.device.write(f"CURR {current}")

    def read_current(self) -> float:
        """
        Read the current from the electronic load.
        :return: Current in Amperes.
        """
        return float(self.device.ask("MEAS:CURR?"))

    def read_voltage(self) -> float:
        """
        Read the voltage from the electronic load.
        :return: Voltage in Volts.
        """
        return float(self.device.ask("MEAS:VOLT?"))

    def error_status(self) -> tuple[int, str]:
        """
        Get the error status of the electronic load.
        :return: Error code and message.
        """
        error = self.device.ask("SYST:ERR?")
        return int(error.split(",")[0]), error.split(",")[1].strip('"')

    def ask_check(self, command: str, num: int = -1) -> str:
        """
        Send a command to the electronic load and check for errors.
        :param command: Command to send.
        :return: Response from the electronic load.
        """
        try:
            response = self.device.ask(command, num)
        except USBError:
            pass
        error = self.error_status()
        if error[0] != 0:
            print(f"{Fore.RED}{command} -> {error}{Fore.RESET}")
            self.load_disable()
            raise ValueError
        return response

    def write_check(self, command: str) -> None:
        """
        Send a command to the electronic load and check for errors.
        :param command: Command to send.
        """
        self.device.write(command)
        error = self.error_status()
        if error[0] != 0:
            print(f"{Fore.RED}{command} -> {error}{Fore.RESET}")
            self.load_disable()
            raise ValueError

    def trace_init(self) -> None:
        """
        Initialize the trace for the electronic load.
        """

        self.ask_check("TRAC:FREE?")
        self.write_check("TRAC:FEED TWO")  # voltage and current
        self.write_check("TRAC:FILTER OFF")  # Disable filter

        # print(self.write_check("TRAC:POIN 10")) # 500 points for voltage and current each
        self.write_check("TRAC:FEED:CONT NEXT")  # NEXT mode, requires clear after each read

        # self.device.write("TRAC:CLE")

    def trace_read(self) -> list:
        """
        Read the trace from the electronic load.
        :return: List of tuples (voltage, current).
        """
        # self.write_check("TRAC:FEED:CONT NEXT")  # Set to NEXT mode
        # self.write_check("TRAC:CLE")  # Set to NEXT mode
        data = self.ask_check("TRAC:DATA?")
        
        # data = self.ask_check("TRAC:DATA?", 1000)
        # Convert to list of tuples (voltage, current)
        data = data.decode().strip().split(",")
        data = [(float(data[i]), float(data[i+1])) for i in range(0, len(data), 2)]
        return data


if __name__ == "__main__":
    eload = Keithley2380(vid=0x05e6, pid=0x2380)
    # eload.trace_init()
    eload.set_current(1.0)
    eload.load_enable()
    times = [0]*1000
    curr = [0]*1000
    volt = [0]*1000
    start = time.time()
    end = time.time()
    i = 0
    while end - start < 3:
        end = time.time()
        eload.read_voltage()
        eload.read_current()
        times[i] = end - start
        i += 1
    

    # ic(eload.trace_read())
    eload.load_disable()
    # print(len(curr))
    
    print(times[5] - times[4])
    # plt.figure(figsize=(10, 5))
    # plt.plot(times, curr, label="Current (A)", color='blue')
    # plt.plot(times, volt, label="Voltage (V)", color='red')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Value")
    # plt.title("Electronic Load Measurements")
    # plt.legend()
    # plt.grid()
    # plt.show()
