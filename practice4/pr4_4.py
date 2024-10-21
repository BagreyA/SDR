import numpy as np
import matplotlib.pyplot as plt

T = 2       
f1 = 1 / T  
k_values = range(1, 6)  

t = np.linspace(0, T, 1000)  

def calculate_integral(k, n):
    s_k = np.sin(2 * np.pi * k * f1 * t)
    s_n = np.sin(2 * np.pi * n * f1 * t)
    integral = np.trapz(s_k * s_n, t)
    return integral

orthogonality_results = {}
for k in k_values:
    for n in k_values:
        integral_value = calculate_integral(k, n)
        orthogonality_results[(k, n)] = integral_value

print("Результаты проверки ортогональности (исходные частоты):")
print("{:<10} {:<10}".format("k", "n", "Integral"))
for k in k_values:
    for n in k_values:
        print(f"R({k}, {n}) = {orthogonality_results[(k, n)]:.4f}")

changed_k = 3
f_changed = 2 * f1

orthogonality_results_changed = {}
for k in k_values:
    for n in k_values:
        if k == changed_k:
            s_k = np.sin(2 * np.pi * changed_k * f_changed * t)
        else:
            s_k = np.sin(2 * np.pi * k * f1 * t)
        s_n = np.sin(2 * np.pi * n * f1 * t)
        integral_value = np.trapz(s_k * s_n, t)
        orthogonality_results_changed[(k, n)] = integral_value
print("\nРезультаты проверки ортогональности (измененные частоты):")
print("{:<10} {:<10}".format("k", "n", "Integral"))
for k in k_values:
    for n in k_values:
        print(f"R({k}, {n}) = {orthogonality_results_changed[(k, n)]:.4f}")

plt.figure(figsize=(12, 8))
for k in k_values:
    if k == changed_k:
        s_k = np.sin(2 * np.pi * changed_k * f_changed * t)
        plt.plot(t, s_k, label=f's_{changed_k}(t) = sin(2π{f_changed:.1f}t)', linestyle='--')
    else:
        s_k = np.sin(2 * np.pi * k * f1 * t)
        plt.plot(t, s_k, label=f's_{k}(t) = sin(2π{f1:.1f}t)')

plt.title('Сигналы s_k(t) для k=1 до 5 с измененной частотой s_3(t)')
plt.xlabel('Время (t)')
plt.ylabel('Амплитуда')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()

