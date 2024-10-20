import numpy as np
import matplotlib.pyplot as plt

# Задаем значения синфазной и квадратурной компонент
in_phase = [1, 0, -1, 1, -1]
quadrature = [-1, -1, 0, 0, 1]

# Определяем частоту и количество графиков
omega = 5e3
num_graphs = len(quadrature)

# Вычисляем фазовый сдвиг для каждой квадратурной компоненты
phase_shift = np.arctan2(quadrature, 1)

# Задаем размер фигуры
plt.figure(figsize=(10, 15))

# Построение первого набора графиков (зеленая кривая)
for idx in range(num_graphs):
    time_values = np.linspace(idx * 1e-3, (idx + 1) * 1e-3, 1000)
    signal = in_phase[idx] * np.cos(2 * np.pi * omega * time_values) - quadrature[idx] * np.sin(2 * np.pi * omega * time_values)
    
    plt.subplot(num_graphs, 3, 3 * idx + 1)
    plt.plot(time_values, signal, color='green')
    plt.title(f'Plot 1 - Oscillation {idx + 1}')
    plt.grid(True)

# Построение второго набора графиков (красная кривая)
for idx in range(num_graphs):
    time_values = np.linspace(idx * 1e-3, (idx + 1) * 1e-3, 1000)
    amplitude = np.sqrt(quadrature[idx]**2 + in_phase[idx]**2)
    signal = amplitude * np.cos(2 * np.pi * omega * time_values + phase_shift[idx])
    
    plt.subplot(num_graphs, 3, 3 * idx + 2)
    plt.plot(time_values, signal, color='red')
    plt.title(f'Plot 2 - Oscillation {idx + 1}')
    plt.grid(True)

# Построение третьего набора графиков (синяя кривая)
for idx in range(num_graphs):
    time_values = np.linspace(idx * 1e-3, (idx + 1) * 1e-3, 1000)
    complex_signal = in_phase[idx] + 1j * quadrature[idx]
    signal = np.real(complex_signal * np.exp(2j * np.pi * omega * time_values))
    
    plt.subplot(num_graphs, 3, 3 * idx + 3)
    plt.plot(time_values, signal, color='blue')
    plt.title(f'Plot 3 - Oscillation {idx + 1}')
    plt.grid(True)

# Компоновка и отображение графиков
plt.tight_layout()
plt.show()
