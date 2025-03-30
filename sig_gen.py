import usbtmc

vid = 0x0957
pid = 0x1607
connection_string = f"USB::0x{vid:04x}::0x{pid:04x}::INSTR"

device = usbtmc.Instrument(connection_string)
print(device.ask("*IDN?"))  # Print the ID of the device