import numpy as np
import matplotlib.pyplot as plt
import commpy.modulation as cm
import scipy.signal
import argparse
import sys
import numpy as np
from scipy.io import savemat, loadmat

class OFDM:
    def __init__(self, modulation_type='QPSK', num_subcarriers=64, cp_length=16, pilot_spacing=16, pilot_value=1-1j, visualize=True, threshold=0.75):
        self._setup_modulation(modulation_type)
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.pilot_spacing = pilot_spacing
        self.pilot_value = pilot_value
        self.visualize = visualize
        self.threshold = threshold
        
        self.pilot_positions = self._generate_pilot_positions()

    def _setup_modulation(self, modulation_type):
        m = None
        if modulation_type == 'QPSK':
            m = 4
            self.modulation = cm.PSKModem(m)
        elif modulation_type == 'QAM1024':
            m = 1024
            self.modulation = cm.QAMModem(m)
        elif modulation_type == 'QAM256':
            m = 256
            self.modulation = cm.QAMModem(m)
        elif modulation_type == 'QAM64':
            m = 64
            self.modulation = cm.QAMModem(m)
        elif modulation_type == 'QAM16':
            m = 16
            self.modulation = cm.QAMModem(m)
        elif modulation_type == '8PSK':
            m = 8
            self.modulation = cm.PSKModem(m)
        elif modulation_type == 'BPSK':
            m = 2
            self.modulation = cm.PSKModem(m)
        else:
            raise ValueError(f"Unsupported modulation type: {modulation_type}")
        self.bits_per_symbol = int(np.log2(m))

    def _generate_pilot_positions(self):
        positions = []
        for i in range(0, self.num_subcarriers, self.pilot_spacing):
            positions.append(i)
        return positions

    def interleave_bits(self, bits, block_rows=2, block_cols=3):
        block_size = block_rows * block_cols
        if len(bits) % block_size != 0:
            raise ValueError("Длина битов должна быть кратна размеру блока")
        
        interleaved = []
        for i in range(0, len(bits), block_size):
            block = bits[i:i+block_size]
            matrix = np.array(block).reshape(block_rows, block_cols)
            for col in range(block_cols):
                interleaved.extend(matrix[:, col].flatten())
    
        return np.array(interleaved)
    
    def deinterleave_bits(self, bits, block_rows=2, block_cols=3):
        block_size = block_rows * block_cols
        if len(bits) % block_size != 0:
            raise ValueError("Длина битов должна быть кратна размеру блока")
        
        deinterleaved = []
        for i in range(0, len(bits), block_size):
            block = bits[i:i+block_size]
            matrix = np.array(block).reshape(block_cols, block_rows)
            for col in range(block_rows):
                deinterleaved.extend(matrix[:, col].flatten())
        
        return np.array(deinterleaved)

    def generate_ofdm_symbol(self, bits):
        print(len(bits))
        interleaved_bits = self.interleave_bits(bits, block_rows=2, block_cols=4)
        max_bits = self.get_max_bits()
        if len(interleaved_bits) < max_bits:
            interleaved_bits = np.resize(interleaved_bits, max_bits)
        data_symbols = self.modulate_bits(interleaved_bits)
        subcarriers = np.zeros(self.num_subcarriers, dtype=complex)
        
        for pos in self.pilot_positions:
            subcarriers[pos] = self.pilot_value
        
        data_indices = [i for i in range(self.num_subcarriers) if i not in self.pilot_positions]
        if len(data_symbols) > len(data_indices):
            raise ValueError("Слишком много данных для доступных поднесущих")
        subcarriers[data_indices[:len(data_symbols)]] = data_symbols
        
        time_domain = np.fft.ifft(subcarriers)
        cp = time_domain[-self.cp_length:]
        ofdm_symbol = np.concatenate([cp, time_domain])
        
        return ofdm_symbol

    def modulate_bits(self, bits):
        remainder = len(bits) % self.bits_per_symbol
        if remainder != 0:
            padding = self.bits_per_symbol - remainder
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        return self.modulation.modulate(bits)

    def _visualize_all(self, subcarriers, ofdm_symbol):
        if not self.visualize:
            return

        plt.figure(figsize=(12, 12))

        plt.subplot(3, 1, 1)
        data_indices = np.array([i for i in range(self.num_subcarriers) if i not in self.pilot_positions])
        data = subcarriers[data_indices]
        data = data[np.abs(data) > 1e-6]

        if len(data) == 0:
            print("Warning: No data points to plot in _visualize_all.")
        else:
            plt.scatter(data.real, data.imag, s=10, label='Данные')
            plt.title(f'Констелляция ({self.modulation.__class__.__name__})')
            plt.xlabel('I (In-Phase)')
            plt.ylabel('Q (Quadrature)')
            plt.grid(True)
            plt.legend()

        plt.subplot(3, 1, 2)
        plt.stem(np.abs(np.fft.fft(subcarriers)), markerfmt='C0o', basefmt='C0-')
        plt.title('Частотная область OFDM-символа')
        plt.xlabel('Поднесущие')
        plt.ylabel('Амплитуда')

        for pos in self.pilot_positions:
            plt.axvline(x=pos, color='r', linestyle='--')

        # 3-й график — временная область OFDM-символа
        plt.subplot(3, 1, 3)
        total_length = len(ofdm_symbol)
        cp_end = self.cp_length
        data_end = total_length

        plt.plot(ofdm_symbol[:cp_end].real, label='Циклический префикс (CP)', color='green')
        plt.plot(ofdm_symbol[:cp_end].imag, color='green', linestyle='--')

        plt.plot(range(cp_end, data_end), ofdm_symbol[cp_end:].real, label='Основной символ', color='blue')
        plt.plot(range(cp_end, data_end), ofdm_symbol[cp_end:].imag, color='blue', linestyle='--')

        plt.title('Временная область OFDM-символа')
        plt.xlabel('Отсчет')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.tight_layout()
        plt.savefig('visualize_all.png')
        plt.close()


    def get_max_bits(self):
        data_subcarriers = self.num_subcarriers - len(self.pilot_positions)
        return data_subcarriers * self.bits_per_symbol

    def find_local_maxima(self, arr, min_distance):
        maxima = []
        for i in range(len(arr)):
            if (i == 0 or arr[i] > arr[i - 1]) and (i == len(arr) - 1 or arr[i] > arr[i + 1]):
                if not maxima or (i - maxima[-1] >= min_distance):
                    maxima.append(i)
        return np.array(maxima)

    def sync_correlation(self, received_signal):
        symbol_length = self.num_subcarriers + self.cp_length
        crosscorr = []
        for i in range(len(received_signal) - symbol_length + 1):
            cp_start = i
            cp_end = i + self.cp_length
            main_start = i + self.num_subcarriers
            main_end = main_start + self.cp_length
            if (cp_end <= len(received_signal)) and (main_end <= len(received_signal)):
                cp_segment = received_signal[cp_start:cp_end]
                main_segment = received_signal[main_start:main_end]
                cc = np.abs(np.correlate(cp_segment / np.linalg.norm(cp_segment),
                                        main_segment / np.linalg.norm(main_segment)))
                crosscorr.append(cc[0])
            else:
                crosscorr.append(0)
        crosscorr = np.array(crosscorr)       
        peaks = self.find_local_maxima(crosscorr, symbol_length)
        
        threshold_value = self.threshold * np.max(crosscorr)
        peaks = [p for p in peaks if crosscorr[p] >= threshold_value]

        valid_peaks = []
        for peak in peaks:
            best_offset = 0
            best_score = -np.inf
            for offset in range(-5, 6):
                candidate_peak = peak + offset
                if candidate_peak < 0 or candidate_peak + symbol_length > len(received_signal):
                    continue
                symbol_with_cp = received_signal[candidate_peak:candidate_peak + symbol_length]
                symbol = symbol_with_cp[self.cp_length:]
                freq_domain = np.fft.fft(symbol)
                pilots = freq_domain[self.pilot_positions]

                weights = np.abs(self.pilot_value)
                current_score = np.sum(weights * np.abs(pilots / self.pilot_value))
                if current_score > best_score:
                    best_score = current_score
                    best_offset = offset
                    
            adaptive_threshold = np.median(np.abs(pilots))
            if best_score > adaptive_threshold:
                valid_peaks.append(peak + best_offset)
        
        if self.visualize:
            self._visualize_sync(received_signal, crosscorr, valid_peaks, peaks)
        
        print(peaks)
        print(valid_peaks)

        return peaks

    def estimate_cfo_from_cp(self, symbol_with_cp):
        cp_length = self.cp_length
        num_subcarriers = self.num_subcarriers
        cp_segment = symbol_with_cp[:cp_length]
        main_segment = symbol_with_cp[cp_length:cp_length*2]
        correlation = np.sum(np.conjugate(cp_segment) * main_segment)
        epsilon = np.angle(correlation) / (2 * np.pi)
        return epsilon

    def compensate_frequency_offset(self, symbol_with_cp, epsilon):
        n = np.arange(len(symbol_with_cp))
        correction = np.exp(-1j * 2 * np.pi * epsilon * n / self.num_subcarriers)
        return symbol_with_cp * correction

    def demodulate(self, received_signal):
        valid_peaks = self.sync_correlation(received_signal)
        symbol_length = self.num_subcarriers + self.cp_length
        all_demod_bits = []
        all_data = []

        for start_index in valid_peaks:
            if start_index + symbol_length > len(received_signal):
                continue
            symbol_with_cp = received_signal[start_index:start_index + symbol_length]
            symbol = symbol_with_cp[self.cp_length:]
            freq_domain = np.fft.fft(symbol)

            pilots = freq_domain[self.pilot_positions]
            data_indices = [i for i in range(self.num_subcarriers) if i not in self.pilot_positions]
            data = freq_domain[data_indices]
            
            H_p = pilots / self.pilot_value
            
            pilot_indices = np.array(self.pilot_positions)
            all_indices = np.arange(self.num_subcarriers)
            real_H = np.real(H_p)
            imag_H = np.imag(H_p)

            f_real = scipy.interpolate.interp1d(pilot_indices, real_H, kind='linear', fill_value="extrapolate")
            f_imag = scipy.interpolate.interp1d(pilot_indices, imag_H, kind='linear', fill_value="extrapolate")
            H_est = f_real(all_indices) + 1j * f_imag(all_indices)

            data_corrected = data / H_est[data_indices]

            demod_bits = self.modulation.demodulate(data_corrected, demod_type='hard').astype(int)

            demod_bits = self.deinterleave_bits(demod_bits, block_rows=2, block_cols=4)

            all_demod_bits.extend(demod_bits)
            
            if self.visualize:
                all_data.append({
                    'raw_data': data,
                    'corrected_data': data_corrected,
                    'H_est': H_est,
                })
        
        if self.visualize:
            self._plot_channel_estimation(all_data)
    
        return np.array(all_demod_bits)

    def _plot_channel_estimation(self, all_data):
        if not self.visualize or not all_data:
            return
        
        first_symbol = all_data[0]
        H_est = first_symbol['H_est']
        
        plt.figure(figsize=(6, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(np.real(H_est), label='Real(H_est)', color='blue')
        plt.plot(np.imag(H_est), label='Imag(H_est)', color='orange')
        plt.scatter(self.pilot_positions, np.real(H_est[self.pilot_positions]), 
                    marker='x', color='red', label='Пилоты')
        plt.title('Оцененный канал H_est')
        plt.xlabel('Поднесущая')
        plt.ylabel('Значение')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.scatter(first_symbol['raw_data'].real, first_symbol['raw_data'].imag, 
                    label='Исходные данные', s=40, alpha=0.5)
        plt.scatter(first_symbol['corrected_data'].real, first_symbol['corrected_data'].imag, 
                    label='Исправленные данные', s=10, alpha=0.5)
        plt.title('Данные до и после компенсации канала')
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('channel_estimation.png')
        plt.close()


    def _visualize_sync(self, received_signal, crosscorr, valid_peaks, all_peaks=None):
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(crosscorr, label='Кросскорреляция')
        plt.scatter(valid_peaks, crosscorr[valid_peaks], color='green', label='Валидные пики')
        if all_peaks is not None:
            plt.scatter(all_peaks, crosscorr[all_peaks], color='red', alpha=0.3, label='Все пики')
        plt.title('Кросскорреляционный анализ')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(received_signal.real, label='I')
        plt.plot(received_signal.imag, label='Q')
        for peak in valid_peaks:
            plt.axvline(x=peak, color='green', linestyle='--')
        if all_peaks is not None:
            for peak in all_peaks:
                plt.axvline(x=peak, color='red', linestyle=':', alpha=0.3)
        plt.title('Синхронизация')
        plt.legend()
        plt.tight_layout()
        plt.savefig('sync_correlation.png')
        plt.close()

    def __str__(self):
        return (f"OFDM Modulator ("
                f"Modulation: {self.modulation.__class__.__name__}, M={self.modulation.M}, "
                f"Subcarriers: {self.num_subcarriers}, "
                f"CP Length: {self.cp_length}, "
                f"Pilot Spacing: {self.pilot_spacing})")

def text_to_bits(text, encoding='utf-8'):
    bytes_data = text.encode(encoding)
    bits = []
    for byte in bytes_data:
        bits.extend([int(bit) for bit in format(byte, '08b')])
    return bits

def bits_to_text(bits, encoding='utf-8'):
    if len(bits) % 8 != 0:
        bits += [0] * (8 - len(bits) % 8)
    bytes_data = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        byte_str = ''.join(map(str, byte_bits))
        bytes_data.append(int(byte_str, 2))
    try:
        return bytes(bytes_data).decode(encoding)
    except UnicodeDecodeError:
        return bytes(bytes_data).decode(encoding, errors='replace')

def read_iq_file(filename, binary=True):
    if binary:
        dt = np.dtype([('real', np.float32), ('imag', np.float32)])
        data = np.fromfile(filename, dtype=dt)
        return data['real'] + 1j*data['imag']
    else:
        with open(filename, 'r') as f:
            return np.array([complex(line.strip()) for line in f])

def write_iq_file(filename, samples, binary=True):
    if binary:
        real = np.real(samples).astype(np.float32)
        imag = np.imag(samples).astype(np.float32)
        with open(filename, 'wb') as f:
            for r, i in zip(real, imag):
                f.write(r.tobytes())
                f.write(i.tobytes())
    else:
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(f"{sample.real},{sample.imag}\n")

def process_modem(args):
    cp_length = args.cp if args.cp is not None else args.subcarriers // 2
    ofdm = OFDM(
        modulation_type=args.modulation,
        num_subcarriers=args.subcarriers,
        cp_length=cp_length,
        pilot_spacing=args.pilot_spacing,
        visualize=True,
        threshold=0.70
    )

    if args.command == 'modulate':
        with open(args.input, 'r') as f:
            text = f.read()
        bits = text_to_bits(text)

        symbols = []
        max_bits = ofdm.get_max_bits()
        count = 0
        for i in range(0, len(bits), max_bits):
            chunk = bits[i:i+max_bits]
            symbols.append(ofdm.generate_ofdm_symbol(chunk))
            count += 1

        all_samples = np.concatenate(symbols)

        write_iq_file(args.output, all_samples, args.binary)

        print(f"{count} OFDM symbols")
        print(f"Modulated to {args.output}")

    elif args.command == 'demodulate':
        samples = read_iq_file(args.input, args.binary)
        
        demod_bits = ofdm.demodulate(samples)
        text = bits_to_text(demod_bits)

        with open(args.output, 'w') as f:
            f.write(text)
        print(f"Demodulated to {args.output}")

def test():
    modulation_type = 'QAM16'
    num_subcarriers = 64
    cp_length = num_subcarriers // 2
    pilot_spacing = num_subcarriers // 32
    pilot_value = 1 + 1j
    ofdm = OFDM(
        modulation_type=modulation_type,
        num_subcarriers=num_subcarriers,
        cp_length=cp_length,
        pilot_spacing=pilot_spacing,
        pilot_value=pilot_value,
        visualize=True,
        threshold=0.70
    )
    max_bits_per_symbol = ofdm.get_max_bits()

    num_symbols = 3
    symbols = []
    all_bits = []
    for _ in range(num_symbols):
        bits = np.random.randint(0, 2, max_bits_per_symbol)
        all_bits.append(bits)
        iq_samples = ofdm.generate_ofdm_symbol(bits)
        symbols.append(iq_samples)
    symbol_length = len(symbols[0]) 
    buffer_length = 20480
    noise = (np.random.randn(buffer_length) + 1j*np.random.randn(buffer_length)) * 0.001
    received_signal = np.copy(noise)
    
    start_in_buffer = 2000
    for symbol in symbols:
        symbol_end = start_in_buffer + len(symbol)
        received_signal[start_in_buffer:symbol_end] += symbol
        start_in_buffer += symbol_length
    
    noise_kernel = np.random.normal(loc=0, scale=1, size=3)
    received_signal = scipy.signal.convolve(received_signal, noise_kernel, mode='full')
    demod_bits = ofdm.demodulate(received_signal)

    all_bits = np.concatenate(all_bits)
    correct = np.sum(all_bits == demod_bits[:len(all_bits)]) / len(all_bits)
    print(f"Точность: {correct * 100:.2f}%")

    for i in range(num_symbols):
        start = i * max_bits_per_symbol
        end = (i+1) * max_bits_per_symbol
        print(f"Символ {i+1}:")
        print("  Исходные биты: ", all_bits[start:end][:10], "...")
        print("  Демодулированные: ", demod_bits[start:end][:10], "...")

def main():

    parser = argparse.ArgumentParser(description="OFDM Modulation/Demodulation")
    subparsers = parser.add_subparsers(dest='command')
    test_parser = subparsers.add_parser('test', help="Run original test scenario")

    mod_parser = subparsers.add_parser('modulate', help="Modulate text file to IQ samples")
    mod_parser.add_argument('-i', '--input', required=True, help="Input text file")
    mod_parser.add_argument('-o', '--output', required=True, help="Output IQ file")
    mod_parser.add_argument('--binary', action='store_true', help="Use binary format")
    mod_parser.add_argument('--modulation', default='QAM16', help="Modulation type")
    mod_parser.add_argument('--subcarriers', type=int, default=2048, help="Number of subcarriers")
    mod_parser.add_argument('--cp', type=int, help="CP length (default: subcarriers//2)")
    mod_parser.add_argument('--pilot-spacing', type=int, default=16, help="Pilot spacing")

    demod_parser = subparsers.add_parser('demodulate', help="Demodulate IQ samples to text")
    demod_parser.add_argument('-i', '--input', required=True, help="Input IQ file")
    demod_parser.add_argument('-o', '--output', required=True, help="Output text file")
    demod_parser.add_argument('--binary', action='store_true', help="Use binary format")
    demod_parser.add_argument('--modulation', default='QAM16', help="Modulation type")
    demod_parser.add_argument('--subcarriers', type=int, default=2048, help="Number of subcarriers")
    demod_parser.add_argument('--cp', type=int, help="CP length (default: subcarriers//2)")
    demod_parser.add_argument('--pilot-spacing', type=int, default=16, help="Pilot spacing")

    args = parser.parse_args()

    if args.command == 'test':
        test()
    elif args.command in ['modulate', 'demodulate']:
        process_modem(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()