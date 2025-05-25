import numpy as np
import random
import scipy.signal
import struct
import matplotlib.pyplot as plt

nFreqSamples = 2048
pilotDistanceInSamples = 32
pilotAmplitude = 2
nData = 256
nCyclic = int(nFreqSamples * 2 / 4)
symbolDuration = 1e-3
samplingRate = nFreqSamples / symbolDuration
subcarrierSpacing = samplingRate / nFreqSamples
k_start = 1

def encode(signal, data, randomSeed=1):
    spectrum = np.zeros(nFreqSamples, dtype=complex)
    k = k_start
    random.seed(randomSeed)
    pilot_counter = pilotDistanceInSamples / 2

    for x in range(nData):
        databyte = int(data[x])
        r = random.randint(0, 255)
        databyte ^= r

        bitstream = np.zeros(8)
        for bit in range(8):
            m = 1 << bit
            bitstream[bit] = 1 if (databyte & m) else -1

        for cnum in range(4):
            pilot_counter -= 1
            if pilot_counter <= 0:
                spectrum[k] = pilotAmplitude
                k += 1
                pilot_counter = pilotDistanceInSamples
                if k >= nFreqSamples:
                    k = 0

            spectrum[k] = complex(bitstream[int(cnum*2)], bitstream[int(cnum*2+1)])
            k += 1
            if k >= nFreqSamples:
                k = 0

    complex_symbol = np.fft.ifft(spectrum)
    tx_symbol = np.zeros(len(complex_symbol)*2)
    s = 1
    txindex = 0

    for smpl in complex_symbol:
        tx_symbol[txindex] = s * np.real(smpl)
        txindex += 1
        tx_symbol[txindex] = s * np.imag(smpl)
        txindex += 1
        s *= -1

    cyclicPrefix = tx_symbol[-nCyclic:]
    signal = np.concatenate((signal, cyclicPrefix, tx_symbol))
    return signal, spectrum

def decode(signal, offset, randomSeed=1):
    rxindex = offset + nCyclic
    rx_symbol = np.zeros(nFreqSamples, dtype=complex)
    s = 1

    for a in range(nFreqSamples):
        realpart = s * signal[rxindex]
        rxindex += 1
        imagpart = s * signal[rxindex]
        rxindex += 1
        rx_symbol[a] = complex(realpart, imagpart)
        s *= -1

    isymbol = np.fft.fft(rx_symbol)
    random.seed(randomSeed)
    k = k_start
    pilot_counter = pilotDistanceInSamples / 2
    data = np.zeros(nData)
    imPilots = 0

    constellation = []

    for x in range(nData):
        bitstream = np.zeros(8)
        for cnum in range(4):
            pilot_counter -= 1
            if pilot_counter <= 0:
                pilot_counter = pilotDistanceInSamples
                imPilots += np.abs(np.imag(isymbol[k]))
                k += 1
                if k >= nFreqSamples:
                    k = 0

            constellation.append((np.real(isymbol[k]), np.imag(isymbol[k])))
            real_bit = 1 if np.real(isymbol[k]) > 0 else 0
            imag_bit = 1 if np.imag(isymbol[k]) > 0 else 0
            bitstream[int(cnum*2)] = real_bit
            bitstream[int(cnum*2+1)] = imag_bit
            k += 1
            if k >= nFreqSamples:
                k = 0

        databyte = 0
        for bit in range(8):
            if bitstream[bit] > 0:
                databyte |= (1 << bit)

        r = random.randint(0, 255)
        databyte ^= r
        data[x] = databyte

    return data.astype(np.uint8), imPilots, constellation, isymbol

def prepare_data_for_transmission(text, message_type=0x01):
    text_bytes = text.encode('utf-8')
    total_length = len(text_bytes)

    length_header = struct.pack('>I', total_length)
    type_header = struct.pack('B', message_type)

    header = length_header + type_header
    data_with_header = header + text_bytes

    chunks = []
    for i in range(0, len(data_with_header), nData):
        chunk = data_with_header[i:i + nData]
        if len(chunk) < nData:
            chunk += b'\x00' * (nData - len(chunk))
        chunks.append(chunk)

    return chunks

def decode_message(data_chunks):
    all_data = b''.join(data_chunks)
    length_header = all_data[:4]
    type_header = all_data[4:5]

    total_length = struct.unpack('>I', length_header)[0]
    message_type = struct.unpack('B', type_header)[0]

    payload = all_data[5:5 + total_length]

    return payload.decode('utf-8', errors='ignore'), message_type

def findSymbolStartIndex(signal, searchrangecoarse=None, searchrangefine=25):
    if not searchrangecoarse:
        searchrangecoarse = nFreqSamples * 10

    max_index = len(signal) - (nFreqSamples * 2 + nCyclic)
    if max_index <= 0:
        raise ValueError("Signal is too short for OFDM symbol detection.")

    crosscorr = []
    for i in range(min(searchrangecoarse, max_index)):
        s1 = signal[i:i + nCyclic]
        s2 = signal[i + nFreqSamples * 2:i + nFreqSamples * 2 + nCyclic]
        if len(s1) > 0 and len(s2) > 0:
            cc = np.correlate(s1, s2)
            crosscorr.append(cc[0] if cc.size > 0 else 0)
        else:
            crosscorr.append(0)

    peaks, _ = scipy.signal.find_peaks(crosscorr, distance=nFreqSamples * 2)
    o1 = peaks[0] if len(peaks) > 0 else 0

    imagpilots = []
    for i in range(max(0, o1 - searchrangefine), min(o1 + searchrangefine, max_index)):
        _, im, _, _ = decode(signal, i)
        imagpilots.append(im)

    best_idx = np.argmin(imagpilots)
    o2 = o1 - searchrangefine + best_idx
    return crosscorr, imagpilots, o2

def visualize_resource_grid(resource_grid, title="Resource Grid"):
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(resource_grid), aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label="Амплитуда")
    plt.title(title)
    plt.xlabel("Поднесущие")
    plt.ylabel("Символы времени")
    plt.show()

def visualize_old_plots(crosscorr, imagpilots, offset, constellation):
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    constellation = np.array(constellation)
    plt.scatter(constellation[:, 0], constellation[:, 1], alpha=0.5)
    plt.title("Constellation Diagram")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()

    plt.subplot(132)
    plt.plot(crosscorr)
    plt.axvline(x=offset, color='r', linestyle='--', label="Detected Symbol Start")
    plt.title("Cross-Correlation for Cyclic Prefix")
    plt.xlabel("Sample Index")
    plt.ylabel("Correlation")
    plt.legend()

    plt.subplot(133)
    plt.plot(range(len(imagpilots)), imagpilots)
    plt.title("Pilot Imaginary Parts for Fine Synchronization")
    plt.xlabel("Relative Sample Index")
    plt.ylabel("Sum of Imaginary Parts")

    plt.tight_layout()
    plt.show()

def validate_parameters():
    nPilots = nFreqSamples // pilotDistanceInSamples
    max_nData = (nFreqSamples - nPilots) // 4

    if nData > max_nData:
        raise ValueError(f"Значение nData ({nData}) слишком велико. Максимальное допустимое значение: {max_nData}.")

def main():
    validate_parameters()
    input_file = "input.txt" 
    output_file = "output.txt"

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Файл {input_file} не найден.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    print(f"Исходный текст из файла:\n{text}")

    chunks = prepare_data_for_transmission(text, message_type=0x01)

    signal = np.zeros(nFreqSamples * 3)
    transmitted_resource_grid = []
    received_resource_grid = []
    all_constellation = []

    for chunk in chunks:
        signal, spectrum = encode(signal, np.frombuffer(chunk, dtype=np.uint8))
        transmitted_resource_grid.append(spectrum)

    signal = np.append(signal, np.zeros(nFreqSamples * 2))

    crosscorr, imagpilots, offset = findSymbolStartIndex(signal)

    received_chunks = []
    symbols_needed = len(chunks)
    for _ in range(symbols_needed):
        data, _, constellation, received_spectrum = decode(signal, offset)
        received_chunks.append(bytes(data))
        all_constellation.extend(constellation)
        received_resource_grid.append(received_spectrum)

    received_text, message_type = decode_message(received_chunks)

    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(received_text)
        print(f"Декодированный текст успешно записан в файл: {output_file}")
    except Exception as e:
        print(f"Ошибка при записи в файл: {e}")

    transmitted_resource_grid = np.array(transmitted_resource_grid).T
    received_resource_grid = np.array(received_resource_grid).T

    print("Ресурсная сетка переданного сигнала:")
    visualize_resource_grid(transmitted_resource_grid, title="Transmitted Resource Grid")

    print("Ресурсная сетка принятого сигнала:")
    visualize_resource_grid(received_resource_grid, title="Received Resource Grid")

    visualize_old_plots(crosscorr, imagpilots, offset, all_constellation)

if __name__ == "__main__":
    main()