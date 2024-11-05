import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import max_len_seq

fs = 1000000
rs = 100000
ns = fs // rs

data = max_len_seq(8)[0]
data = np.concatenate((data, np.zeros(1)))

m = 2 * data - 1

x = np.reshape(m, (2, 128))
xi = x[0, :]
xq = x[1, :]
x_bb = (xi + 1j * xq) / np.sqrt(2)

plt.figure(1)
plt.scatter(x_bb.real, x_bb.imag)
plt.title('Transmitted QPSK Signal')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.grid()

xiq = 2**14 * x_bb

xrec = x_bb + (np.random.normal(0, 0.1, x_bb.shape) + 1j * np.random.normal(0, 0.1, x_bb.shape))

plt.figure(2)
plt.scatter(xrec.real, xrec.imag)
plt.title('Received QPSK Signal (with noise)')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.grid()

plt.show()