#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "my_config.h"
#include "my_init.h"
#include "my_utils.h"


// #define CHRONO
// #define MEMCPY


// DEFAULT PATHs
std::string config_path = "config.cfg";
std::string binar_path = "qpsk_signal.bin";
bool enable_rx = true;
bool enable_tx = true;
std::atomic<bool> keep_running(true);

size_t tx_iters = 0;  // 0 - inf
size_t rx_iters = 0;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        keep_running = false;
        std::cout << "Received SIGINT" << std::endl;
    }
}

void arg_process(std::unordered_map<std::string, std::string> args) {
    if (args.find("c") != args.end()) {
        config_path = args["c"];
    }

    if (args.find("b") != args.end()) {
        binar_path = args["b"];
    }

    if (args.find("rx_only") != args.end()) {
        enable_tx = false;
    }

    if (args.find("tx_only") != args.end()) {
        enable_rx = false;
    }

    if (args.find("tx_iters") != args.end()) {
        tx_iters = std::stoi(args["tx_iters"]);
    }

    if (args.find("rx_iters") != args.end()) {
        rx_iters = std::stoi(args["rx_iters"]);
    }

    if (args.find("iters") != args.end()) {
        rx_iters = std::stoi(args["iters"]);
        tx_iters = rx_iters;
    }
}

void transmit(PlutoDevices &devices, const int16_t *tx_i, const int16_t *tx_q,
              size_t num_samples) {
    std::cout << "TX is run" << std::endl;
    const struct iio_block *txblock;

    size_t _counter = 0;
    size_t iterations = 0;
    #ifdef MEMCPY
        size_t block_size = devices.buffer_size;
        int16_t *prepared_data = new int16_t[block_size];

        // std::memset(prepared_data, 0, block_size * sizeof(int16_t));
        
        size_t samples_to_copy = std::min(num_samples * 2, block_size);
        for (size_t i = 0; i < samples_to_copy / 2; ++i) {
            if (i < num_samples) {
                prepared_data[2 * i] = tx_i[i];     
                prepared_data[2 * i + 1] = tx_q[i];
            }
            else{
                prepared_data[2 * i] = 0;
                prepared_data[2 * i + 1] = 0; 
            }
        }
        
    #endif

    #ifdef CHRONO
        auto start_tx = std::chrono::high_resolution_clock::now();
    #endif

    #ifdef MEMCPY
        while (keep_running && (tx_iters == 0 || iterations < tx_iters)) {
            txblock = iio_stream_get_next_block(devices.txstream);
            int16_t *p_dat = static_cast<int16_t *>(iio_block_first(txblock, devices.tx0_i));
            int16_t *p_end = static_cast<int16_t *>(iio_block_end(txblock));

            size_t available_samples = (p_end - p_dat) / (devices.tx_sample_sz / sizeof(int16_t));
            memcpy(p_dat, prepared_data, block_size * sizeof(int16_t));

            iterations++;
        }

        delete[] prepared_data;
    #else
        while (keep_running && (tx_iters == 0 || iterations < tx_iters)) {
            txblock = iio_stream_get_next_block(devices.txstream);
            int16_t *p_dat;
            ptrdiff_t p_inc = devices.tx_sample_sz;
            int16_t *p_end = static_cast<int16_t *>(iio_block_end(txblock));
            for (p_dat = static_cast<int16_t *>(
                    iio_block_first(txblock, devices.tx0_i));
                p_dat < p_end; p_dat += p_inc / sizeof(*p_dat)) {
                if (_counter < num_samples) {
                    p_dat[0] = tx_i[_counter];  // I
                    p_dat[1] = tx_q[_counter];  // Q
                    _counter++;
                } else {
                    // _counter = 0;
                    p_dat[0] = 0;
                    p_dat[1] = 0;
                }
            }
            iterations++;
        }
    #endif
    std::cout << "TX is down" << std::endl;
    #ifdef CHRONO
        auto end_tx = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_tx = end_tx - start_tx;
        std::cout << "TX time: " << duration_tx.count() << " sec" << std::endl;
    #endif
}

void receive(PlutoDevices &devices, std::ofstream &outfile) {
    std::cout << "RX is run" << std::endl;
    const struct iio_block *rxblock;
    
    #ifdef MEMCPY
        size_t iterations = 0;
        #ifdef CHRONO
            auto start_rx = std::chrono::high_resolution_clock::now();
        #endif
        int16_t *rx_data = new int16_t[devices.buffer_size];
        while (keep_running && (rx_iters == 0 || iterations < rx_iters)) {
            rxblock = iio_stream_get_next_block(devices.rxstream);
            int16_t *p_dat = static_cast<int16_t *>(iio_block_first(rxblock, devices.rx0_i));
            int16_t *p_end = static_cast<int16_t *>(iio_block_end(rxblock));

            size_t num_samples = (p_end - p_dat) / (devices.rx_sample_sz / sizeof(int16_t));
            size_t num_pairs = num_samples / 2; // I/Q

            // int16_t *rx_data = new int16_t[num_samples];

            memcpy(rx_data, p_dat, num_samples * sizeof(int16_t));
            for (size_t j = 0; j < num_pairs; j++) {
                outfile << rx_data[2 * j] << ", " << rx_data[2 * j + 1] << std::endl; // I и Q
            }
           
            iterations++;
        }
        delete[] rx_data;
    #else
       

        std::vector<int16_t> rx_data_i;
        std::vector<int16_t> rx_data_q;
        size_t iterations = 0;
        #ifdef CHRONO
            auto start_rx = std::chrono::high_resolution_clock::now();
        #endif
        while (keep_running && (rx_iters == 0 || iterations < rx_iters)) {
            rxblock = iio_stream_get_next_block(devices.rxstream);
            int16_t *p_dat;
            ptrdiff_t p_inc = devices.rx_sample_sz;
            int16_t *p_end = static_cast<int16_t *>(iio_block_end(rxblock));
            for (p_dat = static_cast<int16_t *>(
                    iio_block_first(rxblock, devices.rx0_i));
                p_dat < p_end; p_dat += p_inc / sizeof(*p_dat)) {
                rx_data_i.push_back(p_dat[0]);  // I
                rx_data_q.push_back(p_dat[1]);  // Q
            }

            for (size_t j = 0; j < rx_data_i.size(); j++) {
                outfile << rx_data_i[j] << ", " << rx_data_q[j] << std::endl;
            }

            rx_data_i.clear();
            rx_data_q.clear();
            iterations++;
        }
    #endif
        std::cout << "RX is down" << std::endl;
        #ifdef CHRONO
            auto end_rx = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_rx = end_rx - start_rx;
            std::cout << "RX time: " << duration_rx.count() << " sec" << std::endl;
        #endif
}

int main(int argc, char *argv[]) {
    std::signal(SIGINT, signal_handler);

    auto args = parse_args(argc, argv);
    arg_process(args);

    Config config;
    if (!load_config(config_path, config)) {
        std::cerr << "Failed to load configuration" << std::endl;
        return 1;
    }

    PlutoDevices devices;
    if (!init_adalm_pluto(config.device_address, config.txcfg, config.rxcfg,
                          devices, config.buffer_size,
                          config.gain_mode.c_str())) {
        std::cerr << "Failed to initialize ADALM-Pluto" << std::endl;
        return 1;
    }
    std::cout << "ADALM-Pluto initialized successfully!" << std::endl;

    int16_t *tx_i = NULL;
    int16_t *tx_q = NULL;
    size_t num_samples = 0;

    if (!read_qpsk_signal(binar_path.c_str(), &tx_i, &tx_q, &num_samples)) {
        std::cerr << "Failed read - " << binar_path << std::endl;
        return 1;
    }
    // upscale
    for (size_t i = 0; i < num_samples; i++) {
        // tx_i[i] *= 512;
        // tx_q[i] *= 512;
        // tx_i[i] *= 32;
        // tx_q[i] *= 32;
        tx_i[i] *= config.multiplier;
        tx_q[i] *= config.multiplier;
        // tx_i[i] = 31999;
        // tx_q[i] = 31999;
    }

    std::ofstream outfile("adalm_rx.txt", std::ios::out);

    std::thread tx_thread;
    std::thread rx_thread;

    if (enable_tx) {

        tx_thread =
            std::thread(transmit, std::ref(devices), tx_i, tx_q, num_samples);
    }
    if (enable_rx) {
        rx_thread = std::thread(receive, std::ref(devices), std::ref(outfile));
    }

    if (enable_tx) {
        tx_thread.join();
    }
    if (enable_rx) {
        rx_thread.join();
    }

    free(tx_i);
    free(tx_q);
    outfile.close();
    cleanup_adalm_pluto(devices);
    free_config(config);

    return 0;
}
