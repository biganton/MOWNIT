import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def tan_f(x=1):
    return np.tan(x)


def d_f(f, h, x=1):
    return (f(x + h) - f(x)) / h


def real_d_f(f, x=1):
    return 1 + f(x) ** 2


def find_maximum(f, domain):
    res = minimize_scalar(lambda x: -f(x), bounds=domain, method='bounded')
    max_value = -res.fun
    return max_value


h_values = np.logspace(0, -16, num=17, base=10)
rounding_errors = []
truncation_errors = []
computational_errors = []
x0 = 1

for h in h_values:
    rounding_errors.append(2 * np.finfo(float).eps / h)
    truncation_errors.append(find_maximum(np.tan, (x0 - h, x0 + h)) * h / 2)
    computational_errors.append(abs(real_d_f(np.tan) - d_f(np.tan, h)))



# Plotting the errors
plt.figure(figsize=(10, 6))
plt.loglog(h_values, rounding_errors, label='Rounding Errors', marker='o')
plt.loglog(h_values, truncation_errors, label='Truncation Errors', marker='^')
plt.loglog(h_values, computational_errors, label='Computational Errors', marker='s')
plt.xlabel('h value')
plt.ylabel('Error magnitude')
plt.title('Rounding, Truncation, and Computational Errors')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()