import numpy as np
import matplotlib.pyplot as plt

f_m = 300
f_c = 10 * f_m
f_s = 2 * f_c
t_s = 1 / 2*f_c
t = np.linspace(0, 0.2, int(t_s)) 

S_t = (2 * np.cos(2 * np.pi * f_m * t) * np.cos(2 * np.pi * f_c * t) -
       np.sin(2 * np.pi * f_m * t) * np.sin(2 * np.pi * f_c * t))

J = 2 * np.cos(2 * np.pi * f_m * t)
Q = np.sin(2 * np.pi * f_m * t)
z = J + 1j * Q

modulus_z = np.abs(z)
arg_z = np.arctan2(Q, J)

envelope_S = np.abs(S_t)

plt.figure(figsize=(12, 6))

plt.plot(t, S_t, label='S(t)', color='blue', alpha=0.6)
plt.plot(t, envelope_S, label='Огибающая |S(t)|', color='red')
plt.plot(t, modulus_z, label='|z|', color='orange')


fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(t, S_t, label='S(t)', color='blue', alpha=0.6)
ax1.plot(t, envelope_S, label='Огибающая |S(t)|', color='red')
ax1.plot(t, modulus_z, label='|z|', color='orange')

ax2 = ax1.twinx()
ax2.plot(t, arg_z, label='arg(z)', color='green', linestyle='-.')
ax2.set_ylabel('arg(z) (радианы)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

ax1.set_title('Сигнал S(t) и его огибающая')
ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Амплитуда')
ax1.grid()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xlim(0, 0.02)
plt.tight_layout()
plt.show()