import numpy as np
import matplotlib.pyplot as plt

def bits_to_qpsk(bits):
    symbols = []
    for i in range(0, len(bits), 2):
        if bits[i] == 0 and bits[i + 1] == 0:
            symbols.append(1 + 1j)  # фаза 00
        elif bits[i] == 0 and bits[i + 1] == 1:
            symbols.append(-1 + 1j)  # фаза 01
        elif bits[i] == 1 and bits[i + 1] == 0:
            symbols.append(-1 - 1j)  # фаза 10
        elif bits[i] == 1 and bits[i + 1] == 1:
            symbols.append(1 - 1j)  # фаза 11
    return np.array(symbols)

def matching_filter(signal, filter_kernel):
    return np.convolve(signal, filter_kernel, mode='same')

def gardner_sync(signal, mu=0.01):
    n = len(signal)
    delta = 0
    sync_signal = []
    for i in range(1, n-1):
        error = np.imag(signal[i]) - np.imag(signal[i-1]) 
        delta += mu * error
        synced_sample = signal[i + int(delta)]
        sync_signal.append(synced_sample)
    return np.array(sync_signal)

def plot_signals_and_constellation(before_filter, after_filter, after_gardner, label="Signal"):
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 2, 1)
    plt.plot(np.real(before_filter), label="R-part Before Filter")
    plt.title("R-part Before Matching Filter")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(np.real(after_filter), label="R-part After Filter")
    plt.title("R-part After Matching Filter")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(np.imag(before_filter), label="J-part Before Filter")
    plt.title("J-part Before Matching Filter")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(np.imag(after_filter), label="J-part After Filter")
    plt.title("J-part After Matching Filter")
    plt.grid(True)
    plt.legend()

    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase horizontal and vertical spacing between subplots

    plt.figure(figsize=(6, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(np.real(after_filter), np.imag(after_filter), label="After Filter", color='blue')
    plt.title("Constellation After Matching Filter")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(np.real(after_gardner), np.imag(after_gardner), label="After Gardner", color='red')
    plt.title("Constellation After Gardner")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.legend()

    plt.show()

def plot_rectangular_pulse_shape(R_before, R_after, J_before, J_after):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(R_before, label="R-part Before Filter", color='blue')
    plt.title("R part After Matching filter")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(R_after, label="R-part After Filter", color='red')
    plt.title("R part After Gardner")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(J_before, label="J-part Before Filter", color='green')
    plt.title("J part After Matching filter")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(J_after, label="J-part After Filter", color='orange')
    plt.title("J part After Gardner")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def simulate_qpsk():

    bit_sequence = np.random.randint(0, 2, 200)

    qpsk_symbols = bits_to_qpsk(bit_sequence)

    noise = np.random.normal(0, 0.1, qpsk_symbols.shape) + 1j * np.random.normal(0, 0.1, qpsk_symbols.shape)
    qpsk_symbols_noisy = qpsk_symbols + noise

    filter_kernel = np.array([1, 4, 6, 4, 1]) / 16
    filtered_signal = matching_filter(qpsk_symbols_noisy, filter_kernel)

    gardner_signal = gardner_sync(filtered_signal)

    # Разделяем сигнал на R- и J-части
    R_before = np.real(qpsk_symbols_noisy)
    J_before = np.imag(qpsk_symbols_noisy)

    R_after = np.real(filtered_signal)
    J_after = np.imag(filtered_signal)

    # Визуализация сигналов
    plot_rectangular_pulse_shape(R_before, R_after, J_before, J_after)

    # Визуализация остальных графиков
    plot_signals_and_constellation(qpsk_symbols_noisy, filtered_signal, gardner_signal)

# Запуск симуляции QPSK и фильтрации
simulate_qpsk()
