#include "my_utils.h"

void read_qpsk_signal(const char *filename, int16_t **tx_i, int16_t **tx_q, size_t *num_samples) {
    // Открываем бинарный файл для чтения
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Ошибка открытия файла");
        exit(EXIT_FAILURE);
    }

    // Определяем размер файла
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Проверяем, что размер файла четный
    if (file_size % (2 * sizeof(int16_t)) != 0) {
        fprintf(stderr, "Неверный размер файла\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Вычисляем количество пар (I, Q)
    *num_samples = file_size / (2 * sizeof(int16_t));

    *tx_i = (int16_t *)malloc(*num_samples * sizeof(int16_t));
    *tx_q = (int16_t *)malloc(*num_samples * sizeof(int16_t));
    if (*tx_i == NULL || *tx_q == NULL) {
        perror("Ошибка выделения памяти");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < *num_samples; i++) {
        fread(&(*tx_i)[i], sizeof(int16_t), 1, file);
        fread(&(*tx_q)[i], sizeof(int16_t), 1, file);
    }

    fclose(file);
}