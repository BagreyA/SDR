import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

T = 2       
tau = 0.5   
f0 = 1 / T  
Ts = 0.001  
N_periods = 3   

t = np.arange(0, N_periods * T, Ts)

x_t = signal.square(2 * np.pi * f0 * t, duty=tau/T)

def calc_an(n):
    cos_comp = np.cos(2 * np.pi * n * f0 * t)
    return (1 / T) * np.sum(x_t * cos_comp) * Ts

def calc_bn(n):
    sin_comp = np.sin(2 * np.pi * n * f0 * t)
    return (1 / T) * np.sum(x_t * sin_comp) * Ts

# Вычисление коэффициентов для n = 0, 1, 2, 3, 4, 5, 6
a0 = (1 / T) * np.sum(x_t) * Ts
a_values = [a0] + [calc_an(n) for n in range(1, 7)]
b_values = [0] + [calc_bn(n) for n in range(1, 7)]  

def synthesize_signal(n_terms, t):
    y = np.zeros_like(t)
    for n in range(n_terms + 1):
        a_n = a_values[n]
        b_n = b_values[n]
        y += a_n * np.cos(2 * np.pi * n * f0 * t) + b_n * np.sin(2 * np.pi * n * f0 * t)
    return y

synthesized_signal_2 = synthesize_signal(2, t)
synthesized_signal_4 = synthesize_signal(4, t)
synthesized_signal_6 = synthesize_signal(6, t)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, synthesized_signal_2, label='Сумма до 2 гармоники')
plt.title('Синтез временного колебания (до 2 гармоники)')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, N_periods * T)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, synthesized_signal_4, label='Сумма до 4 гармоники', color='orange')
plt.title('Синтез временного колебания (до 4 гармоники)')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, N_periods * T)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, synthesized_signal_6, label='Сумма до 6 гармоники', color='green')
plt.title('Синтез временного колебания (до 6 гармоники)')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, N_periods * T)
plt.legend()

plt.tight_layout()
plt.show()
