import numpy as np
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider

def bits_to_qpsk(bits):
    symbols = []
    for i in range(0, len(bits), 2):
        if i + 1 < len(bits):
            if bits[i] == 0 and bits[i + 1] == 0:
                symbols.append(1 + 1j)  # 00
            elif bits[i] == 0 and bits[i + 1] == 1:
                symbols.append(-1 + 1j)  # 01
            elif bits[i] == 1 and bits[i + 1] == 0:
                symbols.append(-1 - 1j)  # 10
            elif bits[i] == 1 and bits[i + 1] == 1:
                symbols.append(1 - 1j)  # 11
    return np.array(symbols)

def convolve_with_filter(symbols, filter_coefficients):
    result = np.convolve(symbols, filter_coefficients, mode='full')
    return result

def extract_every_nth(signal, n, offset=0):
    return signal[offset::n]


def bit_sequence_to_text(bit_sequence):
    # if len(bit_sequence) % 8 != 0:
    #     raise ValueError("Длина битовой последовательности должна быть кратна 8.")
    
    text = []

    for i in range(0, len(bit_sequence), 8):
        byte = bit_sequence[i:i + 8]  # Берем 8 бит
        ascii_value = int(''.join(str(bit) for bit in byte), 2) 
        text.append(chr(ascii_value))
    
    return ''.join(text)

def plot_eye_diagram(signal, sps, num_traces=100):
    plt.figure(figsize=(12, 6))
    plt.title('Eye Diagram')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    for i in range(num_traces):
        start_idx = i * sps
        if start_idx + 2 * sps > len(signal):
            break
        plt.plot(range(2 * sps), signal[start_idx:start_idx + 2 * sps].real, color='blue', alpha=0.5)

    plt.show()

def plot_signal_interactive(signal, N):
    fig_real_imag, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    real_line, = ax_real.plot([], [], color='blue', label='Real Part')
    imag_line, = ax_imag.plot([], [], color='red', label='Imaginary Part')

    for ax in [ax_real, ax_imag]:
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.legend()

    ax_real.set_title('Real Part')
    ax_real.set_xlabel('Index')
    ax_real.set_ylabel('Amplitude')

    ax_imag.set_title('Imaginary Part')
    ax_imag.set_xlabel('Index')
    ax_imag.set_ylabel('Amplitude')

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
    scatter = ax_scatter.scatter([], [], s=5)
    ax_scatter.set_title('QPSK Constellation')
    ax_scatter.set_xlabel('I')
    ax_scatter.set_ylabel('Q')
    ax_scatter.grid(True)
    ax_scatter.axis('equal')

    def update_graph(offset):
        offset = int(offset)
        extracted_signal = extract_every_nth(signal, N, offset)
        real_part = extracted_signal.real
        imag_part = extracted_signal.imag

        real_line.set_data(range(len(real_part)), real_part)
        imag_line.set_data(range(len(imag_part)), imag_part)

        scatter.set_offsets(np.c_[real_part, imag_part])

        ax_real.set_xlim(0, len(real_part))
        ax_real.set_ylim(min(real_part), max(real_part))

        ax_imag.set_xlim(0, len(imag_part))
        ax_imag.set_ylim(min(imag_part), max(imag_part))

        ax_scatter.set_xlim(min(real_part), max(real_part))
        ax_scatter.set_ylim(min(imag_part), max(imag_part))

        fig_real_imag.canvas.draw_idle()
        fig_scatter.canvas.draw_idle()

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Offset', 0, N-1, valinit=0, valstep=1)

    slider.on_changed(update_graph)

    update_graph(0)

    plt.show()
    return slider.val

def plot_signal(real_part, imag_part):
    time_indices = range(len(real_part))

    plt.figure(figsize=(12, 6))

    # real
    plt.subplot(2, 1, 1)
    plt.plot(time_indices, real_part, color='blue', label='Real Part')
    # plt.plot(time_indices, imag_part, color='red', label='Imaginary Part')
    plt.title('Real Part')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.legend()

    # img
    plt.subplot(2, 1, 2)
    plt.plot(time_indices, imag_part, color='red', label='Imaginary Part')
    plt.title('Imaginary Part')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.legend()

    # QPSK
    plt.figure()
    plt.scatter(real_part, imag_part, s=5)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

def read_signal(filename):
    signal = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            # signal.append([np.int16(parts[0]), np.int16(parts[1])])
            signal.append(np.complex128(np.int16(parts[0]) + 1j * np.int16(parts[1])))
    return np.array(signal)


def qpsk_to_bits(symbols):
    bits = []
    for symbol in symbols:
        real = np.real(symbol)
        imag = np.imag(symbol)
        if real > 0 and imag > 0:
            bits.extend([0, 0])  # 1 + 1j
        elif real < 0 and imag > 0:
            bits.extend([0, 1])  # -1 + 1j
        elif real < 0 and imag < 0:
            bits.extend([1, 0])  # -1 - 1j
        elif real > 0 and imag < 0:
            bits.extend([1, 1])  # 1 - 1j
    return np.array(bits)


# Barker sequence
def barker_sequence():
    return np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)

# Frequency synchronization
def frequency_sync(signal, pilot, fs, oversample_factor):
    """
    Частотная коррекция по известному пилотному сигналу.
    """
    pilot_mapped = 2 * pilot - 1  # [0, 1] -> [-1, 1]
    correlation = correlate(signal, pilot_mapped, mode='same')
    phase_offset = np.angle(correlation)  # Фазовый сдвиг
    time = np.arange(len(signal)) / fs
    freq_offset = np.mean(np.diff(phase_offset) / (2 * np.pi * time[1]))

    print(f"Оцененное частотное смещение: {freq_offset} Гц")

    # Коррекция частоты
    corrected_signal = signal * np.exp(-1j * 2 * np.pi * freq_offset * time)

    # Оверсэмплинг — оставляем только каждый N-й символ
    corrected_signal = corrected_signal[::oversample_factor]
    return corrected_signal

# Time synchronization
def time_sync(signal, pilot, threshold=0.9):
    """
    Поиск временной синхронизации с использованием пиков корреляции.
    """
    pilot_mapped = 2 * pilot - 1  # [0, 1] -> [-1, 1]
    correlation = correlate(signal, pilot_mapped, mode='full')
    correlation /= np.max(np.abs(correlation))  # Нормализация

    # Поиск пиков выше порога
    peaks, _ = find_peaks(correlation, height=threshold)
    if len(peaks) > 0:
        sync_index = peaks[0] - len(pilot) + 1  # Коррекция смещения корреляции
        print(f"Синхронизация найдена на индексе: {sync_index}")
        return sync_index
    print("Синхронизация не найдена")
    return -1

def find_sync_word(signal, sync_word, threshold=0.9):
    sync_word_mapped = 2 * sync_word - 1  # [0, 1] -> [-1, 1]
    signal_mapped = 2 * signal - 1        # [0, 1] -> [-1, 1]
    # print(signal)
    correlation = correlate(signal_mapped, sync_word_mapped, mode='valid')
    norm_correlation = correlation / (len(sync_word) * np.mean(np.abs(sync_word_mapped)))

    max_corr = np.max(norm_correlation)
    max_index = np.argmax(norm_correlation)
    # print(max_corr)
    if max_corr > threshold:
        # print(max_corr)
        # print(sync_word_mapped)
        # print(signal_mapped)
        return max_index
    return -1


# Sliding window for packet extraction
def sliding_window(signal, sync_word, data_length, threshold=0.8):
    sync_word_len = len(sync_word)
    signal_len = len(signal)
    packets = []
    index = 0

    while index <= signal_len - sync_word_len:
        sync_index = find_sync_word(signal[index:(sync_word_len + index)], sync_word, threshold)

        if sync_index != -1:
            print(f"Sync word found at index {index + sync_index}")

            data_start = index + sync_index + sync_word_len
            data_end = data_start + data_length - sync_word_len

            if data_end > signal_len:
                print("Incomplete data detected, stopping extraction.")
                break

            data_bits = signal[data_start:data_end]
            packets.append(data_bits)
            index = data_end
        else:
            index += 1

    return packets

# Read input
parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
args = parser.parse_args()

# Parameters
N = 10  # Oversampling factor

# Read and preprocess the signal
signal = read_signal(args.file_path)
filter_ones = np.ones(N)
signal = convolve_with_filter(signal, filter_ones)

# Interactive plotting to extract sample index
# n = plot_signal_interactive(signal, N)
# signal = extract_every_nth(signal, N, n)
signal = extract_every_nth(signal, N, N-1)
plot_signal(signal.real, signal.imag)
# Normalize the signal
signal /= np.mean(np.abs(signal))

# Frequency synchronization
fs = 1.2e9  # Sampling frequency (update as necessary)
sync_word = barker_sequence()
signal = frequency_sync(signal, sync_word, fs, N)
plot_signal(signal.real, signal.imag)
# Time synchronization
start_index = time_sync(signal, sync_word, threshold=0.9)
if start_index == -1:
    raise ValueError("Синхронизация не выполнена")

# Crop signal based on synchronization index
signal = signal[start_index:]

# Decode signal
bits = qpsk_to_bits(signal)

data_length = 198  # Maximum packet size
packets = sliding_window(bits, sync_word, data_length, threshold=0.7)
print(f"Found {len(packets)} packet(s) in the signal.")

for i, payload_bits in enumerate(packets, start=1):
    text = bit_sequence_to_text(payload_bits)
    print(f"Packet {i}: {text}")