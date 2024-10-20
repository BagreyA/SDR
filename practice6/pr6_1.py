import numpy as np
import matplotlib.pyplot as plt

# Создаем временной массив
t = np.linspace(0, 5 * 10 ** (-3), 1000)
i = [1, 0, -1, 1, -1]
q = [-1, -1, 0, 0, 1]

omega = 5 * 10 ** 3
count = 0
count1 = 1

# Вычисляем фазовый угол
fi = np.arctan2(q, 1)

# Первый график
for g in range(len(q)):
    t = np.linspace(count * 10 ** (-3), count1 * 10 ** (-3), 1000)
    T = i[g] * np.cos(2 * np.pi * omega * t) - q[g] * np.sin(2 * np.pi * omega * t)
    count += 1
    count1 += 1
    plt.subplot(3, 1, 1)
    plt.plot(t, T, color='green')
    plt.title('График 1')
    plt.grid(True)

count = 0
count1 = 1

# Второй график
for g in range(len(q)):
    t = np.linspace(count * 10 ** (-3), count1 * 10 ** (-3), 1000)
    a = np.sqrt(q[g] ** 2 + i[g] ** 2)
    T = a * np.cos(2 * np.pi * omega * t + fi[g])
    count += 1
    count1 += 1
    plt.subplot(3, 1, 2)
    plt.plot(t, T, color='red')
    plt.title('График 2')
    plt.grid(True)

count = 0
count1 = 1

# Третий график
for g in range(len(q)):
    t = np.linspace(count * 10 ** (-3), count1 * 10 ** (-3), 1000)
    xl = i[g] + 1j * q[g]
    T = np.real(xl * np.exp(2 * np.pi * 1j * omega * t))
    
    count += 1
    count1 += 1
    plt.subplot(3, 1, 3)
    plt.plot(t, T, color='blue')
    plt.title('График 3')
    plt.grid(True)

# Настройка отображения
plt.tight_layout()
plt.show()
