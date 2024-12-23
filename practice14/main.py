import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d

# Параметры системы
Fs = 1000  # Частота дискретизации
symbol_rate = 100  # Скорость символов (бит/с)
samples_per_symbol = Fs // symbol_rate  # Количество отсчетов на символ
frame_size = 100  # Размер фрейма в символах
preamble_size = 20  # Размер преамбулы

# Генерация QPSK сигнала с преамбулой
def generate_qpsk_signal(frame_size, preamble_size, symbol_rate, Fs):
    # Преамбула
    preamble = np.random.randint(0, 4, size=preamble_size)
    preamble_qpsk = 2 * (preamble // 2) - 1 + 1j * (2 * (preamble % 2) - 1)
    preamble_signal = np.repeat(preamble_qpsk, samples_per_symbol)
    
    # Сигналы фрейма
    frame = np.random.randint(0, 4, size=frame_size)
    frame_qpsk = 2 * (frame // 2) - 1 + 1j * (2 * (frame % 2) - 1)
    frame_signal = np.repeat(frame_qpsk, samples_per_symbol)
    
    # Полный сигнал (преамбула + фрейм)
    signal = np.concatenate([preamble_signal, frame_signal])
    return signal

# Генерация сигнала
qpsk_signal = generate_qpsk_signal(frame_size, preamble_size, symbol_rate, Fs)

# Добавление шума
def add_noise(signal, snr_dB):
    snr = 10**(snr_dB / 10)
    power_signal = np.mean(np.abs(signal)**2)
    noise_power = power_signal / snr
    noise = np.sqrt(noise_power) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) / np.sqrt(2)
    return signal + noise

# Добавляем шум в сигнал
snr_dB = 10  # Уровень отношения сигнал/шум
noisy_signal = add_noise(qpsk_signal, snr_dB)

# Визуализация принятого сигнала
plt.figure(figsize=(10, 6))
plt.plot(np.real(noisy_signal), label='Re(s(t))')
plt.plot(np.imag(noisy_signal), label='Im(s(t))')
plt.title('Принятый сигнал с шумом')
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)
plt.show()

# Согласованный фильтр (фильтрация и ресэмплирование)
def matched_filter(signal, pulse_shape):
    return np.convolve(signal, pulse_shape, mode='same')

# Пример фильтра (например, использование sinc-функции как импульсной характеристики)
def generate_pulse_shape(samples_per_symbol):
    t = np.arange(-5, 5, 1 / samples_per_symbol)
    pulse = np.sinc(t)
    return pulse / np.linalg.norm(pulse)

pulse_shape = generate_pulse_shape(samples_per_symbol)
filtered_signal = matched_filter(noisy_signal, pulse_shape)

# Символьная синхронизация методом Гарднера
def gardner_sync(signal, samples_per_symbol):
    error = np.zeros(len(signal))
    d = np.zeros(len(signal))
    
    for i in range(1, len(signal) - 1):
        d[i] = np.angle(signal[i] * np.conj(signal[i - 1]))  # Разница фаз
        error[i] = np.real(d[i])  # Ошибка фазы
    return error

sync_error = gardner_sync(filtered_signal, samples_per_symbol)

# Визуализация ошибки синхронизации
plt.figure(figsize=(10, 6))
plt.plot(sync_error)
plt.title('Ошибка символьной синхронизации (метод Гарднера)')
plt.xlabel('Отсчеты')
plt.ylabel('Ошибка фазы')
plt.grid(True)
plt.show()

# Анализ диаграммы и вывод каждого 10-го отсчета
def plot_signal_diagram(signal, samples_per_symbol):
    plt.figure(figsize=(10, 6))
    real_vals = np.real(signal[::samples_per_symbol])
    imag_vals = np.imag(signal[::samples_per_symbol])
    
    plt.scatter(real_vals, imag_vals, c='b', label='Сигнальные точки', marker='o')
    plt.title('Диаграмма сигнала QPSK')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.legend()
    
    # Добавление меток для каждого символа
    for i in range(len(real_vals)):
        if i % 10 == 0:  # Вывод каждого 10-го отсчета
            plt.text(real_vals[i], imag_vals[i], f'{i}', fontsize=8)
    
    plt.show()

plot_signal_diagram(filtered_signal, samples_per_symbol)

# Ручной выбор наилучшего отсчета
def find_best_sample(sync_error):
    best_index = np.argmin(np.abs(sync_error))
    return best_index

best_sample_index = find_best_sample(sync_error)


# Созвездие с оптимальной точкой
def plot_enhanced_constellation(signal, best_sample_index):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(signal), np.imag(signal), c='g', marker='o', label='Созвездие')
    plt.scatter(np.real(signal[best_sample_index]), np.imag(signal[best_sample_index]), c='r', marker='x')
    plt.title('Созвездие QPSK')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_enhanced_constellation(filtered_signal[::samples_per_symbol], best_sample_index)
