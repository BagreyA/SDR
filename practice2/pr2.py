import adi
import numpy as plt
import matplotlib.pyplot as plt

sdr = adi.Pluto('usb:1.9.5') # or whatever your Pluto's IP is
sdr.sample_rate = int(2.5e6)
sdr.rx_lo = int(830e6)
sdr.tx_lo = int(830e6)
sdr.tx_cyclic_buffer = True

rectangle = []
for i in range(1024):
    if i < 300 or i > 700:
        rectangle.append(complex(0))
    else:
        rectangle.append(complex(4000))

plt.figure(figsize=[10, 5])
plt.plot(rectangle)
plt.show()


big_array = []
sdr.tx(rectangle)
for i in range(5):
    rx = sdr.rx()
    for i in rx:
        big_array.append(abs(i))

plt.figure(figsize=[10, 5])
plt.plot(big_array)
plt.show()