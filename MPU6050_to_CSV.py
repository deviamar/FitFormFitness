import serial
import csv
from datetime import datetime

port = '/dev/cu.usbmodem11201'
baud = 115200
output_file = 'Test_7.csv'

ser = serial.Serial(port, baud)
print(f"Logging data to {output_file}...")

try:
	with open(output_file, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["time", "ax", "ay", "az", "time", "gx", "gy", "gz"])
		while True:
			line = ser.readline().decode('utf-8').strip()
			if line:
				data = line.split(",")
				print(f"Line not empty. Data: {data}")
				writer.writerow(data)

except KeyboardInterrupt:
	print("\nData logging stopped by user.")

finally:
	ser.close()
	print("Serial port closed.")
