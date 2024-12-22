import numpy as np
import matplotlib.pyplot as plt
import argparse  # <-- Add this import

def read_signal(filename):
    # Read the binary data from the file
    signal = np.fromfile(filename, dtype=np.int16)

    # Normalize the signal to the range [-1, 1]
    signal = signal / 32767.0

    # Separate the real and imaginary parts (assuming interleaved data)
    real_part = signal[::2]  # Every other value is the real part
    imag_part = signal[1::2]  # The other values are the imaginary part

    return real_part, imag_part

def low_pass_filter(data, kernel_size):
    kernel = np.ones(kernel_size)
    return np.convolve(data, kernel, mode='same')

def plot_signal(real_part, imag_part):
    time_indices = range(len(real_part))

    plt.figure(figsize=(12, 6))

    # real
    plt.subplot(2, 1, 1)
    plt.plot(time_indices, real_part, color='blue', label='Real Part')
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

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('--max_lines', type=int, default=None)
    args = parser.parse_args()

    real_part, imag_part = read_signal(args.file_path)

    plot_signal(real_part, imag_part)

if __name__ == '__main__':
    main()
