import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

np.set_printoptions(precision=16)


def equation(x_k, x_k_1):
    return 2.25 * x_k - 0.5 * x_k_1


def generate_sequence(x0, x1, n, precision):
    x = np.zeros(n, dtype=precision)
    x[0], x[1] = x0, x1
    for k in range(1, n - 1):
        x[k + 1] = equation(x[k], x[k - 1])
    return x


def xk_real_value(k):
    return np.power(4, -k)/3


x0_single = np.float32(1/3)
x1_single = np.float32(1/12)
x0_double = np.float64(1/3)
x1_double = np.float64(1/12)
x0_fraction = Fraction(1, 3)
x1_fraction = Fraction(1, 12)

n_single = 225
x_single_tab = generate_sequence(x0_single, x1_single, n_single, np.float32)

n_double = 60
x_double_tab = generate_sequence(x0_double, x1_double, n_double, np.float64)

n_fraction = 225
x_fraction_tab = generate_sequence(x0_fraction, x1_fraction, n_fraction, object)

plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(n_single), x_single_tab, label='Single Precision', linestyle='--')
plt.semilogy(np.arange(n_double), x_double_tab, label='Double Precision', linestyle='dashdot')
plt.semilogy(np.arange(n_fraction), [float(x) for x in x_fraction_tab], label='Fractions', linestyle='dotted')
plt.xlabel('k')
plt.ylabel('x[k]')
plt.title('Wartość ciągu w zależności od k')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, n_single), np.abs((x_single_tab[1:] - 4/3) / (4/3)), label='Single Precision')
plt.plot(np.arange(1, n_double), np.abs((x_double_tab[1:] - 4/3) / (4/3)), label='Double Precision')
plt.plot(np.arange(1, n_fraction), np.abs((np.array(x_fraction_tab[1:], dtype=np.float64) - 4/3) / (4/3)), label='Fractions')
plt.xlabel('k')
plt.ylabel('Błąd względny')
plt.title('Błąd względny w zależności od k')
plt.legend()
plt.grid(True)
plt.show()