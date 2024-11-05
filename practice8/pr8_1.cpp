#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <iio.h>
#include <unistd.h>

#define SAMPLE_RATE 1000000
#define TX_LO 2000000000
#define RX_LO 2000000000
#define BUFFER_SIZE 2048

void transmit_signal(struct iio_device *dev, complex double *signal, int num_samples) {
    // Prepare transmit buffer
    int i;
    for (i = 0; i < num_samples; i++) {
        // Scale the signal for transmission
        int16_t sample_real = (int16_t)(creal(signal[i]) * (1 << 14)); // Scale
        int16_t sample_imag = (int16_t)(cimag(signal[i]) * (1 << 14)); // Scale

        // Send data to the SDR
        struct iio_buffer *txbuf = iio_device_get_buffer(dev, num_samples);
        if (!txbuf) {
            perror("Failed to get transmit buffer");
            return;
        }
        
        // Writing to the buffer
        int16_t *ptr = (int16_t *)iio_buffer_start(txbuf);
        ptr[i * 2] = sample_real;
        ptr[i * 2 + 1] = sample_imag;
        
        // Push data to hardware
        iio_buffer_push(txbuf);
    }
}

void receive_signal(struct iio_device *dev, complex double *signal, int num_samples) {
    // Receive data from the SDR
    struct iio_buffer *rxbuf = iio_device_get_buffer(dev, num_samples);
    if (!rxbuf) {
        perror("Failed to get receive buffer");
        return;
    }
    
    // Reading from the buffer
    iio_buffer_refill(rxbuf);
    int16_t *ptr = (int16_t *)iio_buffer_start(rxbuf);
    for (int i = 0; i < num_samples; i++) {
        signal[i] = ptr[i * 2] + I * ptr[i * 2 + 1]; // Create complex signal
    }
    
    // Release the buffer
    iio_buffer_destroy(rxbuf);
}

int main() {
    struct iio_context *ctx = iio_create_context_from_uri("ip:192.168.2.1");
    if (!ctx) {
        perror("Failed to create context");
        return -1;
    }

    // Get the TX and RX devices
    struct iio_device *tx_device = iio_context_find_device(ctx, "pluto_tx");
    struct iio_device *rx_device = iio_context_find_device(ctx, "pluto_rx");
    if (!tx_device || !rx_device) {
        perror("Failed to find TX or RX device");
        return -1;
    }

    // Set up sample rate and frequency
    iio_device_attr_write(tx_device, "frequency", "2000000000");
    iio_device_attr_write(rx_device, "frequency", "2000000000");
    iio_device_attr_write(tx_device, "sample_rate", "1000000");
    iio_device_attr_write(rx_device, "sample_rate", "1000000");

    // Prepare the QPSK signal
    int num_samples = 128;
    complex double *tx_signal = malloc(num_samples * sizeof(complex double));
    
    // Fill tx_signal with QPSK data here (e.g., 1, 1, -1, -1, etc.)
    for (int i = 0; i < num_samples; i++) {
        tx_signal[i] = cos(2 * M_PI * i / num_samples) + I * sin(2 * M_PI * i / num_samples);
    }

    // Transmit signal
    transmit_signal(tx_device, tx_signal, num_samples);

    // Receive signal
    complex double *rx_signal = malloc(num_samples * sizeof(complex double));
    receive_signal(rx_device, rx_signal, num_samples);

    // Process received signal here (e.g., visualization or further analysis)

    // Clean up
    free(tx_signal);
    free(rx_signal);
    iio_context_destroy(ctx);
    return 0;
}
