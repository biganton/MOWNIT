import numpy
import matplotlib

def tan_f(x = 1):
    return np.tan(x)

def d_f(f, h, x = 1):
    return (f(x + h) - f(x))/h

def real_d_f(f, x = 1):
    return 1 + np.tan(x)**2

h_values = np.logspace(0, -16, num=17, base=10)
print(h_values)