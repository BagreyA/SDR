import numpy as np
import matplotlib.pyplot as plt

A = 5          
f = 4           
T = 1 / f       
Ts = 0.01       
phi = 0         

t = np.arange(0, T, Ts)

x_t = A * np.cos(2 * np.pi * f * t + phi)

# Опорные колебания для вычисления коэффициентов ряда Фурье
s_c = np.cos(2 * np.pi * f * t)
s_s = np.sin(2 * np.pi * f * t)

# Вычисление произведений сигнала на опорные колебания
m1 = x_t * s_c
m2 = x_t * s_s

# Интегрирование (в нашем случае - суммирование с учётом шага дискретизации Ts)
a1 = (1 / T) * np.sum(m1) * Ts
a2 = (1 / T) * np.sum(m2) * Ts

print(f'Коэффициент при cos (a1): {a1}')
print(f'Коэффициент при sin (a2): {a2}')

def calc_an(n):
    cos_comp = np.cos(2 * np.pi * n * f * t)
    return (2 / T) * np.sum(x_t * cos_comp) * Ts

def calc_bn(n):
    sin_comp = np.sin(2 * np.pi * n * f * t)
    return (2 / T) * np.sum(x_t * sin_comp) * Ts

a0 = (2 / T) * np.sum(x_t) * Ts

n_values = [0, 1, 2, 3, 4]
a_values = [a0] + [calc_an(n) for n in range(1, 5)]
b_values = [0] + [calc_bn(n) for n in range(1, 5)] 

print("Коэффициенты Фурье:")
for n in range(5):
    print(f"a_{n} = {a_values[n]:.5f}, b_{n} = {b_values[n]:.5f}")
    
    
def calc_an(n):
    cos_comp = np.cos(2 * np.pi * n * f * t)
    return (2 / T) * np.sum(x_t * cos_comp) * Ts

def calc_bn(n):
    sin_comp = np.sin(2 * np.pi * n * f * t)
    return (2 / T) * np.sum(x_t * sin_comp) * Ts

a0 = (2 / T) * np.sum(x_t) * Ts

n_values = [0, 1, 2, 3, 4]
a_values = [a0] + [calc_an(n) for n in range(1, 5)]
b_values = [0] + [calc_bn(n) for n in range(1, 5)]

A_values = [np.sqrt(a_n**2 + b_n**2) for a_n, b_n in zip(a_values, b_values)]
phi_values = [np.arctan2(b_n, a_n) for a_n, b_n in zip(a_values, b_values)]

print("Значения a_n:", a_values)
print("Значения b_n:", b_values)
print("Амплитуды A_n:", A_values)
print("Фазы φ_n (в радианах):", phi_values)

plt.figure(figsize=(10,6))

plt.plot(t, x_t, label='x(t) = A*cos(2πft)')
plt.plot(t, s_c, label='cos(2πft)', linestyle='--')
plt.plot(t, s_s, label='sin(2πft)', linestyle='--')
plt.plot(t, m1, label='x(t)*cos(2πft)', linestyle='-.')
plt.plot(t, m2, label='x(t)*sin(2πft)', linestyle='-.')
plt.ylim(-A, A)
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Гармоническое колебание и произведения для вычисления коэффициентов ряда Фурье')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(t, x_t, label='x(t) = A*cos(2πf0t)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Гармоническое колебание x(t)')
plt.grid(True)
plt.legend()
plt.show()

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