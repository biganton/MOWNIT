import numpy as np
from fractions import Fraction


def equation(x_k, x_k_1):
    return 2.25 * x_k - 0.5 * x_k_1


def generate_sequence(x0, x1, n, precision):
    x = np.zeros(n, dtype=precision)
    x[0], x[1] = x0, x1
    for k in range(1, n - 1):
        x[k + 1] = equation(x[k], x[k - 1])
    return x


x0_single = np.float32(1/3)
x1_single = np.float32(1/12)
x0_double = np.float64(1/3)
x1_double = np.float64(1/12)
x0_fraction = Fraction(1, 3)
x1_fraction = Fraction(1, 12)

n_single = 5
x_single_tab = generate_sequence(x0_single, x1_single, n_single, np.float32)

n_double = 5
x_double_tab = generate_sequence(x0_double, x1_double, n_double, np.float64)

n_fraction = 5
x_fraction_tab = generate_sequence(x0_fraction, x1_fraction, n_fraction, object)

print(x_single_tab)
print(x_double_tab)
print(x_fraction_tab)