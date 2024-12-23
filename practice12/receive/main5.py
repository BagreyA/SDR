from scipy.signal import correlate
import matplotlib.pyplot as plt
import argparse

import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

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

def raised_cosine_filter(beta, sps, num_taps):
    t = np.arange(-num_taps / 2, num_taps / 2 + 1) / sps
    h = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] == 0:
            h[i] = 1 - beta + (4 * beta / np.pi)
        elif abs(t[i]) == 1 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * ((1 + np.cos(np.pi * (4 * beta))) / 2)
        else:
            h[i] = (np.sin(np.pi * t[i] * (1 - beta)) + (4 * beta * t[i]) * np.cos(np.pi * t[i] * (1 + beta))) / (np.pi * t[i] * (1 - (4 * beta * t[i]) ** 2))

    h /= np.sqrt(np.sum(h ** 2))
    return h

def convolve_with_filter(symbols, filter_coefficients):
    result = np.convolve(symbols, filter_coefficients, mode='full')
    return result

def extract_every_nth(signal, n, offset=0):
    return signal[offset::n]

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

def rotate_constellation(signal, ref_points):
    rotations = [np.exp(1j * theta) for theta in np.linspace(0, 2 * np.pi, 360, endpoint=False)]
    
    def alignment_cost(rotation):
        rotated_signal = signal * rotation
        closest_points = np.array([ref_points[np.argmin(np.abs(rotated_signal[i] - ref_points))] for i in range(len(rotated_signal))])
        return np.sum(np.abs(rotated_signal - closest_points))
    
    best_rotation = min(rotations, key=alignment_cost)

    return signal * best_rotation

def costas_loop(signal, loop_gain, init_phase=0):
    phase = init_phase
    output_signal = []
    for sample in signal:
        rotated_sample = sample * np.exp(-1j * phase)
        phase += loop_gain * np.angle(rotated_sample**4)
        output_signal.append(rotated_sample)
    return np.array(output_signal)

def barker_sequence():
    return np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    # return np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.int8)

# def bits_to_bpsk(bits):
#     # BPSK: 0 -> -1, 1 -> +1
#     return 2 * bits - 1

def bits_to_bpsk_complex(bits):
    return np.array((2 * bits - 1) + 0j)

def find_sync_word(signal, sync_word, threshold=0.8):
    sync_word_bpsk = bits_to_bpsk_complex(sync_word)
    sync_word_mapped = 2 * sync_word_bpsk.real - 1
    signal_mapped = 2 * signal.real - 1

    correlation = correlate(signal_mapped, sync_word_mapped, mode='valid')  # Реальная часть
    # correlation = correlate(signal.real, sync_word_bpsk.real, mode='valid')  # Реальная часть
    # norm_correlation = correlation / (len(sync_word) * np.mean(np.abs(sync_word_bpsk)))
    # norm_correlation = correlation / (len(sync_word) * np.mean(np.abs(sync_word_bpsk.real)))

    norm_signal = np.linalg.norm(signal_mapped)
    norm_sync_word = np.linalg.norm(sync_word_mapped)

    # norm_signal = np.linalg.norm(signal)
    # norm_sync_word = np.linalg.norm(sync_word_bpsk)
    

    # Нормализованная корреляция
    norm_correlation = correlation / (norm_signal * norm_sync_word)

    max_corr = np.max(norm_correlation)
    max_index = np.argmax(norm_correlation)
    # print(max_corr)
    # print(max_index)
    if max_corr > threshold:
        print(max_corr) 
        return max_index
    return -1

def sliding_window(signal, sync_word, data_length, threshold=0.8):
    sync_word_len = len(sync_word)
    signal_len = len(signal)
    packets = []
    index = 0
    # print(signal)
    while index <= signal_len - sync_word_len:
        sync_index = find_sync_word(signal[index:(sync_word_len+index)], sync_word, threshold)

        if sync_index != -1:
            print(f"Sync word found at index {index + sync_index}")

            # Найден синхроворд, извлекаем данные
            data_start = index + sync_index + sync_word_len
            data_end = data_start + data_length
            # print(data_start)
            # print(data_end)
            # print(signal_len)
            if data_end > signal_len:
                print("Incomplete data detected, stopping extraction.")
                break
            
            data_symbols = signal[data_start:data_end]
            # print(data_symbols)
            packets.append(data_symbols)
            index = data_end
        else:
            index += 1

    return packets

# def symbol_synchronization(I, Q, Ns, Nsp, BnTs=0.01, damping_factor=1, detector_gain=2.7):
#     p1, p2 = 0, 0
#     r = 0  
#     offsets = [] 
#     Nsps = int(1 / BnTs) 

#     theta = (BnTs / Nsps) / (damping_factor + 0.25 / damping_factor)
#     d = (1 + 2 * damping_factor * theta + theta**2) * detector_gain
#     K1 = (-4 * damping_factor * theta) / d
#     K2 = (-4 * theta**2) / d

#     print(K1)
#     print(K2)

#     for n in range(0, len(I) - Ns, Ns):
#         if n + Nsp + Ns >= len(I) or n + Ns >= len(I):
#             break

#         # TED
#         e = (
#             (I[n + Nsp + Ns] - I[n + Ns]) * I[n + Nsp // 2 + Ns] +
#             (Q[n + Nsp + Ns] - Q[n + Ns]) * Q[n + Nsp // 2 + Ns]
#         )

#         # Фильтр
#         p1 = e * K1
#         p2 += p1 + e * K2
#         if p2 > 1:
#             while p2 > 1:
#                 p2 -= 1

#         r_prev = r
#         r = (r + 1 + p2 * Nsps) % Ns

#         if r < r_prev:
#             offsets.append(round((p2 * Nsps)))
#         else:
#             offsets.append(0)
#     print(offsets)
#     return offsets


def symbol_synchronization(I, Q, Ns, Nsp, BnTs=0.01, damping_factor = 1, detector_gain = 2.7):
    symbols = np.empty(len(I), dtype=np.complex128) 

    p1, p2 = 0, 0 
    r = 0 
    offset = 0
    Nsps = int(1 / BnTs) 
    offsets = [] 

    theta = (BnTs/Nsps)/(damping_factor+0.25/damping_factor)
    d = (1 + 2 * damping_factor * theta + theta**2) * detector_gain
    K1 = (-4*damping_factor*theta)/d
    K2 = (-4*theta**2)/d

    print(K1)
    print(K2)

    # for n in range(0, len(I) - Ns, Ns):
    for n in range(Ns, len(I)-Ns):
        if n + Nsp + Ns >= len(I) or n + Ns >= len(I):
            break

        # TED
        e = (
            (I[n + Nsp + Ns] - I[n + Ns]) * I[n + Nsp // 2 + Ns] +
            (Q[n + Nsp + Ns] - Q[n + Ns]) * Q[n + Nsp // 2 + Ns]
        )
        # e /= 1000
        # фильтр
        p1 = e * K1
        p2 = p2 + p1 + e * K2
        print("--- n:",n, "---")
        print("e", e)
        print("p1", p1)
        print("p2", p2)
        if p2 > 1:
            while p2 > 1:
                print(p2)
                p2 -= 1
        offset = round(p2 * (Nsps))
        # offset = 9
        offsets.append(offset)

        print("r",r)
        r_prev = r
        r = (r_prev + offset*n + 1) % Ns
        # r = (r_prev + 1 + offset) % Ns
        # r += (offset + p2 + e) % Ns
        if r < r_prev:
            # r = 0
            # r_prev = 0
            # symbols[offset] = I[n] + 1j * Q[n]
            n += Ns
            # offset = 0

        
    print(offsets)
    # return symbols
    return offsets

def extract_symbols(I, Q, offsets, Ns):
    symbols = np.empty(len(offsets), dtype=np.complex128) 
    for n, offset in enumerate(offsets):
        index = n * Ns + offset
        if index < len(I) and index < len(Q):
            symbols[n] = I[index] + 1j * Q[index]
            # print(index)
    return symbols

# Приём
parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
args = parser.parse_args()

N = 10
signal = read_signal(args.file_path)

# plot_signal(synchronized_signal.real, synchronized_signal.imag)

filter_ones = np.ones(N)
signal = convolve_with_filter(signal, filter_ones)
# plot_signal(signal.real, signal.imag)
# signal = extract_every_nth(signal, N, N - 1)
offsets = symbol_synchronization(signal.real, signal.imag, N, int(N / 2), 0.1, 1.0, 2.7)
signal = extract_symbols(signal.real, signal.imag, offsets, N)
# signal = symbol_synchronization(signal.real, signal.imag, N, int(N / 2), 0.1, 1, 2.7)
# signal = extract_symbols(signal.real, signal.imag, offsets, N)
# print(symbols)
signal /= N
plot_signal(signal.real, signal.imag)
plot_eye_diagram(signal, 1)
sync_word = barker_sequence()
data_length = 88 # Максимальный размер В СИМВОЛАХ IQ

# Поиск пакетов
packets = sliding_window(signal, sync_word, data_length, threshold=0.1)

# Обработка пакетов
print(f"Found {len(packets)} packet(s) in the signal.")
for i, qpsk_symbols in enumerate(packets, start=1):
    data_bits = qpsk_to_bits(qpsk_symbols)
    # plot_eye_diagram(qpsk_symbols, 1)
    # plot_signal(qpsk_symbols.real, qpsk_symbols.imag)
    # print(data_bits)
    text = bit_sequence_to_text(data_bits)
    print(f"Packet {i}: {text}")