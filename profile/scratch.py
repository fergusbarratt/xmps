from numpy import load, linspace, exp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ts = load('ts.npy')
def f(x, a, b, c):
    return a*x**b+c
def g(x, a, b, c):
    return a*exp(b*x)+c

a, b, c = curve_fit(f, list(range(1, 16)), ts)[0]
a_, b_, c_ = curve_fit(g, list(range(1, 16)), ts)[0]

def f_(x): return f(x, a, b, c)
def g_(x): return g(x, a_, b_, c_)
plt.scatter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], ts, marker='x')
T = linspace(1, 15)
plt.plot(T, f_(T))
print(a, b, c)
plt.title('$\sim D^5$')
plt.savefig('profile/scaling.pdf')
plt.show()
