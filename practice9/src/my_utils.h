#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void read_qpsk_signal(const char *filename, int16_t **tx_i, int16_t **tx_q, size_t *num_samples);

#endif  // UTILS_H