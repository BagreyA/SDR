import numpy as np
import adi
import matplotlib.pyplot as plt

# 1. Генерация бинарной последовательности (bits)
def generate_bits(n_bits):
    return np.random.randint(0, 2, n_bits)

# 2. Модуляция QPSK
def bits_to_qpsk(bits):
    symbols = []
    for i in range(0, len(bits), 2):
        if i + 1 < len(bits):
            if bits[i] == 0 and bits[i + 1] == 0:
                symbols.append(1 + 1j)  # фаза 00
            elif bits[i] == 0 and bits[i + 1] == 1:
                symbols.append(-1 + 1j)  # фаза 01
            elif bits[i] == 1 and bits[i + 1] == 0:
                symbols.append(-1 - 1j)  # фаза 10
            elif bits[i] == 1 and bits[i + 1] == 1:
                symbols.append(1 - 1j)  # фаза 11
    return np.array(symbols)

# 3. Пересэмплирование
def oversample_qpsk(symbols, N):
    oversampled_signal = []
    for symbol in symbols:
        oversampled_signal.append(symbol)  # I
        oversampled_signal.append(0)       # 0 для I
        oversampled_signal.append(0)       # 0 для I
        oversampled_signal.append(0)       # и так далее для Q
    return np.array(oversampled_signal)

# 4. Фильтрация
def filter_signal(signal, filter_kernel):
    if signal.size == 0:
        raise ValueError("Signal is empty.")
    if filter_kernel.size == 0:
        raise ValueError("Filter kernel is empty.")
    return np.convolve(signal, filter_kernel, mode='same')

# 5. Амплификация
def amplify_signal(signal, gain=2**11):
    return signal * gain

# 6. Генерация пилообразного сигнала с уменьшенной частотой
def generate_sawtooth_signal(frequency, sample_rate, duration):
    t = np.arange(0, duration, 1/sample_rate)
    return 2 * (np.mod(frequency * t, 1) - 0.5)  # Пилообразный сигнал

# Настройки SDR
sdr = adi.Pluto('ip:192.168.2.1')
sdr.sample_rate = 1000000
sdr.tx_lo = 2000000000
sdr.rx_lo = 2000000000

# Генерация бинарной последовательности
bit_sequence = generate_bits(200)

# Модуляция QPSK
qpsk_symbols = bits_to_qpsk(bit_sequence)

# 1. Визуализация после модуляции
plt.figure()
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag)
plt.title("QPSK Constellation (After Modulation)")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Пересэмплирование
oversampled_signal = oversample_qpsk(qpsk_symbols, N=10)

# 2. Визуализация после пересэмплирования
plt.figure()
plt.plot(np.real(oversampled_signal), label="Real (I)")
plt.plot(np.imag(oversampled_signal), label="Imaginary (Q)")
plt.title("Oversampled QPSK Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Применение фильтрации
filter_kernel = np.ones(5)  # Простое скользящее окно
filtered_signal = filter_signal(oversampled_signal, filter_kernel)

# 3. Визуализация после фильтрации
plt.figure()
plt.plot(np.real(filtered_signal), label="Real (I)")
plt.plot(np.imag(filtered_signal), label="Imaginary (Q)")
plt.title("Filtered QPSK Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Амплификация
amplified_signal = amplify_signal(filtered_signal)

# 4. Визуализация после амплификации
plt.figure()
plt.plot(np.real(amplified_signal), label="Real (I)")
plt.plot(np.imag(amplified_signal), label="Imaginary (Q)")
plt.title("Amplified QPSK Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Генерация пилообразного сигнала с уменьшенной частотой (помедленнее)
# Уменьшаем частоту пилообразного сигнала на 10 раз
sawtooth_signal = generate_sawtooth_signal(frequency=100, sample_rate=sdr.sample_rate, duration=1)

# Визуализация пилообразного сигнала с уменьшенной частотой
plt.figure()
plt.plot(sawtooth_signal)
plt.title("Sawtooth Signal with Lower Frequency")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Слияние с QPSK сигналом (для создания комбинации)
final_signal = amplified_signal + sawtooth_signal[:len(amplified_signal)]

# Визуализация итогового сигнала
plt.figure()
plt.plot(np.real(final_signal), label="Real (I)")
plt.plot(np.imag(final_signal), label="Imaginary (Q)")
plt.title("Final Signal (QPSK + Sawtooth)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
plt.close()  # Закрыть график

# Подготовка данных для передачи
tx_data = np.zeros(2 * len(final_signal))
tx_data[0::2] = np.real(final_signal)  # I
tx_data[1::2] = np.imag(final_signal)  # Q

# Отправка данных
sdr.tx(tx_data)

# Получение и отображение полученного сигнала
sdr.rx_hardwaregain_chan0 = -5
sdr.rx_buffer_size = len(tx_data)
received_signal = sdr.rx()

# 5. Визуализация после приема
plt.figure()
plt.scatter(received_signal.real, received_signal.imag)
plt.title("Received Signal (QPSK + Sawtooth)")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()
plt.close()  # Закрыть график
