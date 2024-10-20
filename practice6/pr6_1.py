import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры времени и сигнала
time_steps = np.linspace(0, 5e-3, 1000)
in_phase = [1, 0, -1, 1, -1]
quadrature = [-1, -1, 0, 0, 1]
frequency = 5e3

# Определение фазового сдвига
phase_shift = np.arctan2(quadrature, 1)

# Инициализация переменных для временного шага
start_time = 0
end_time = 1

# Построение первого графика (зелёная кривая)
for idx in range(len(quadrature)):
    current_time = np.linspace(start_time * 1e-3, end_time * 1e-3, 1000)
    signal = in_phase[idx] * np.cos(2 * np.pi * frequency * current_time) - quadrature[idx] * np.sin(2 * np.pi * frequency * current_time)
    
    plt.subplot(3, 1, 1)
    plt.plot(current_time, signal, color='green')
    plt.title('Signal Representation 1')
    plt.grid(True)
    
    start_time += 1
    end_time += 1

# Сброс временных переменных
start_time = 0
end_time = 1

# Построение второго графика (красная кривая)
for idx in range(len(quadrature)):
    current_time = np.linspace(start_time * 1e-3, end_time * 1e-3, 1000)
    amplitude = np.sqrt(quadrature[idx]**2 + in_phase[idx]**2)
    signal = amplitude * np.cos(2 * np.pi * frequency * current_time + phase_shift[idx])
    
    plt.subplot(3, 1, 2)
    plt.plot(current_time, signal, color='red')
    plt.title('Signal Representation 2')
    plt.grid(True)
    
    start_time += 1
    end_time += 1

# Сброс временных переменных
start_time = 0
end_time = 1

# Построение третьего графика (синяя кривая)
for idx in range(len(quadrature)):
    current_time = np.linspace(start_time * 1e-3, end_time * 1e-3, 1000)
    complex_signal = in_phase[idx] + 1j * quadrature[idx]
    signal = np.real(complex_signal * np.exp(2j * np.pi * frequency * current_time))
    
    plt.subplot(3, 1, 3)
    plt.plot(current_time, signal, color='blue')
    plt.title('Signal Representation 3')
    plt.grid(True)
    
    start_time += 1
    end_time += 1

# Компоновка и вывод графиков
plt.tight_layout()
plt.show()
