import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

np.set_printoptions(precision=16)


def equation(x_k, x_k_1, a1, a2):
    return a1 * x_k - a2 * x_k_1


def generate_sequence(x0, x1, a1, a2, n):
    x = [x0, x1]
    for k in range(1, n - 1):
        x.append(equation(x[k], x[k - 1], a1, a2))
    return x


def xk_real_value(k):
    return 1 / (4**k * 3)


x0_single = np.float32(1 / 3)
x1_single = np.float32(1 / 12)
x0_double = np.float64(1 / 3)
x1_double = np.float64(1 / 12)
x0_fraction = Fraction(1, 3)
x1_fraction = Fraction(1, 12)
a1 = 2.25
s2 = 0.5

n_single = 225
x_single_tab = generate_sequence(x0_single, x1_single, np.float32(a1), np.float32(s2), n_single)

n_double = 60
x_double_tab = generate_sequence(x0_double, x1_double, np.float64(a1), np.float64(s2), n_double)

n_fraction = 225
x_fraction_tab = generate_sequence(x0_fraction, x1_fraction, Fraction(a1), Fraction(s2), n_fraction)

plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(n_single), x_single_tab, label='Single precision')
plt.semilogy(np.arange(n_double), x_double_tab, label='Double precision')
plt.semilogy(np.arange(n_fraction), x_fraction_tab, label='Fraction precision')
plt.xlabel('k')
plt.ylabel('x[k]')
plt.title('Value of sequence depending on k')
plt.legend()
plt.grid(True)
plt.show()

single_relative_errors = []
double_relative_errors = []
fraction_relative_errors = []

for i in range(255):
    x_real = xk_real_value(i)
    if i < n_single: single_relative_errors.append(abs(x_single_tab[i] - x_real) / x_real)
    if i < n_double: double_relative_errors.append(abs(x_double_tab[i] - x_real) / x_real)
    if i < n_fraction: fraction_relative_errors.append(abs(x_fraction_tab[i] - x_real) / x_real)

plt.figure(figsize=(10, 6))
plt.semilogy(np.arange(n_single), single_relative_errors, label='Single precision')
plt.semilogy(np.arange(n_double), double_relative_errors, label='Double precision')
plt.semilogy(np.arange(n_fraction), fraction_relative_errors, label='Fraction precision')
plt.xlabel('k')
plt.ylabel('Relative error')
plt.title('Relative error depending on k')
plt.legend()
plt.grid(True)
plt.show()