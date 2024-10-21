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

plt.figure(figsize=(8, 4))
plt.plot(t, x_t)
plt.title('Прямоугольный сигнал')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()

def calc_an(n):
    cos_comp = np.cos(2 * np.pi * n * f0 * t)
    return (2 / T) * np.sum(x_t * cos_comp) * Ts

def calc_bn(n):
    sin_comp = np.sin(2 * np.pi * n * f0 * t)
    return (2 / T) * np.sum(x_t * sin_comp) * Ts

a0 = (2 / T) * np.sum(x_t) * Ts

n_values = [0, 1, 2, 3, 4]
a_values = [a0] + [calc_an(n) for n in range(1, 5)]
b_values = [0] + [calc_bn(n) for n in range(1, 5)]  

A_values = [np.sqrt(a_n**2 + b_n**2) for a_n, b_n in zip(a_values, b_values)]
phi_values = [np.arctan2(b_n, a_n) for a_n, b_n in zip(a_values, b_values)]

print("Коэффициенты Фурье и фазы:")
for n in range(5):
    print(f"n = {n}: a_{n} = {a_values[n]:.5f}, b_{n} = {b_values[n]:.5f}, A_{n} = {A_values[n]:.5f}, φ_{n} = {phi_values[n]:.5f} рад")

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(n_values, A_values)
plt.xlabel('Гармоника n')
plt.ylabel('Амплитуда An')
plt.title('Амплитудный спектр An')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(n_values, phi_values)
plt.xlabel('Гармоника n')
plt.ylabel('Фаза φn (рад)')
plt.title('Фазовый спектр φn')
plt.grid(True)

plt.tight_layout()
plt.show()
