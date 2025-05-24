import numpy as np
import matplotlib.pyplot as plt

FILENAME = '../../data/txdata_bark3.pcm'
NSPS = 10
NUM_TRACES = 100

data = np.fromfile(FILENAME, dtype=np.int16).astype(np.float32)
I_raw = data[::2]
Q_raw = data[1::2]

filter_kernel = np.ones(NSPS) / NSPS
I_filtered = np.convolve(I_raw, filter_kernel, mode='full')
Q_filtered = np.convolve(Q_raw, filter_kernel, mode='full')

BnTs = 0.01
Kp = 0.002
zeta = np.sqrt(2) / 2
theta = (BnTs / NSPS) / (zeta + 0.25/zeta)
denom = (1 + 2*zeta*theta + theta**2) * Kp
K1 = -4 * zeta * theta / denom
K2 = -4 * theta**2 / denom

p1, p2, tau = 0.0, 0.0, 0
errors, taus = [], []
sync_I, sync_Q = [], []

for i in range(0, len(I_filtered) - NSPS, NSPS):
    idx_start = i + tau
    idx_mid = idx_start + NSPS // 2
    idx_end = idx_start + NSPS

    if idx_end >= len(I_filtered) or idx_start < 0:
        break

    # Извлекаем отсчеты
    I_s, I_m, I_e = I_filtered[idx_start], I_filtered[idx_mid], I_filtered[idx_end]
    Q_s, Q_m, Q_e = Q_filtered[idx_start], Q_filtered[idx_mid], Q_filtered[idx_end]

    # Алгоритм Гарднера: оценка ошибки
    error = (I_e - I_s) * I_m + (Q_e - Q_s) * Q_m
    errors.append(error)

    # Обновление значений tau с помощью ПИД-регулятора
    p1 = error * K1
    p2 += p1 + error * K2
    p2 = np.clip(p2, -1.0, 1.0)
    tau = int(round(p2 * NSPS))
    taus.append(tau)

    # Сохраняем синхронизированные значения
    sync_I.append(I_m)
    sync_Q.append(Q_m)

plt.figure(figsize=(14, 10))

plt.subplot(221)
plt.scatter(sync_I, sync_Q)
plt.title('Созвездие QPSK после синхронизации')
plt.grid(True)

plt.subplot(222)
plt.plot(errors)
plt.title('Ошибка синхронизации (Gardner TED)')
plt.grid(True)

plt.subplot(223)
plt.plot(taus)
plt.title('Смещение tau (оценка фазы символа)')
plt.grid(True)

plt.subplot(224)
plt.scatter(I_filtered, Q_filtered)
plt.title('Созвездие до синхронизации (после фильтрации)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Обработано символов: {len(sync_I)}")
print(f"Средняя ошибка TED: {np.mean(errors):.4f}")
print(f"Макс. смещение tau: {max(taus):.2f}")
print(f"Среднее значение tau: {np.mean(taus):.4f}")
