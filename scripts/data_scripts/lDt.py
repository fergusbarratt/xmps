from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from numpy import cumsum as cs, arange as ar, linspace as lin, exp
from scipy.optimize import curve_fit
mpl.style.use('ggplot')

ls = load('data/spectra/maxs.npy')
def g(x, a, b):
    return a*x**b
def h(x, a, b):
    return a*exp(b*x)
def f(x): return h(x, *curve_fit(h, [1, 2, 3, 4, 5, 6, 7], ls)[0])
def f_(x): return g(x, *curve_fit(g, [1, 2, 3, 4, 5, 6, 7], ls)[0])

#plt.scatter([1, 2, 3, 4, 5, 6, 7], ls)
#plt.plot(lin(1, 7, 100), f(lin(1, 7, 100)))
#plt.plot(lin(1, 7, 100), f_(lin(1, 7, 100)))
#plt.show()
#f = interp1d([1, 2, 3, 4, 5, 6, 7], ls, kind='cubic')

F = load('data/spectra/Dt.npy')
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(f(F))
ax[0].set_title('$\lambda(D(t))$', loc='right')
ax[1].plot(cs(f(F))*0.01)
ax[1].set_title('$\int^t dt\lambda(D(t))$', loc='right')
ax[2].plot(F)
ax[2].set_title('$D(t)$', loc='right')
plt.tight_layout()
plt.savefig('images/spectra/lDt.pdf')
plt.show()
