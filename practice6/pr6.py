import numpy as np
import matplotlib.pyplot as plt

i = [1, 0, -1, 1, -1]
q = [-1, -1, 0, 0, 1]

omega = 5 * 10 ** 3
n_graphs = len(q)

fi = np.arctan2(q, 1)

plt.figure(figsize=(10, 15))

for g in range(n_graphs):
    t = np.linspace(g * 10 ** (-3), (g + 1) * 10 ** (-3), 1000)
    T = i[g] * np.cos(2 * np.pi * omega * t) - q[g] * np.sin(2 * np.pi * omega * t)
    
    plt.subplot(n_graphs, 3, 3 * g + 1)
    plt.plot(t, T, color='green')
    plt.title(f'График 1 - Колебание {g+1}')
    plt.grid(True)

for g in range(n_graphs):
    t = np.linspace(g * 10 ** (-3), (g + 1) * 10 ** (-3), 1000)
    a = np.sqrt(q[g] ** 2 + i[g] ** 2)
    T = a * np.cos(2 * np.pi * omega * t + fi[g])
    
    plt.subplot(n_graphs, 3, 3 * g + 2)
    plt.plot(t, T, color='red')
    plt.title(f'График 2 - Колебание {g+1}')
    plt.grid(True)

for g in range(n_graphs):
    t = np.linspace(g * 10 ** (-3), (g + 1) * 10 ** (-3), 1000)
    xl = i[g] + 1j * q[g]
    T = np.real(xl * np.exp(2 * np.pi * 1j * omega * t))
    
    plt.subplot(n_graphs, 3, 3 * g + 3)
    plt.plot(t, T, color='blue')
    plt.title(f'График 3 - Колебание {g+1}')
    plt.grid(True)

plt.tight_layout()
plt.show()
