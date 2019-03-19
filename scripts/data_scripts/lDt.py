from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from numpy import cumsum as cs, arange as ar, linspace as lin, exp, log
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
mpl.style.use('ggplot')

ls = []
for D in range(1, 9):
    x = load('data/spectra/run_6/lyapunovs_L6_D{}_N80000.npy'.format(D))
    ls.append(sum(abs(x[-1]))/2)

S = load('data/S.npy')

def g(x, a, b):
    return a*x**b+2
def h(x, a, b):
    return a*exp(b*x)
xs = list(range(1, len(ls)+1))
a, b = curve_fit(h, xs, ls)[0]
print(a, b)
def f(x): return h(x, a, b)
def f_(x): return g(x, *curve_fit(g, xs, ls)[0])
#f__ = interp1d(xs, ls)

#plt.scatter(xs, ls)
#plt.plot(lin(1, xs[-1], 100), f(lin(1, xs[-1], 100)))
#plt.plot(lin(1, xs[-1], 100), f_(lin(1, xs[-1], 100)))
#plt.plot(lin(1, xs[-1], 100), f__(lin(1, xs[-1], 100)))

F = load('data/spectra/Dt.npy')
fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(f(F))
ax[0].set_title('$\lambda(D(t))$', loc='right')
ax[1].plot(cs(f(F))*0.01)
ax[1].set_title('$\int^t dt\lambda(D(t))$', loc='right')
ax[2].plot(F)
ax[2].set_title('$D(t)$', loc='right')
ax[3].plot(log(F))
plt.tight_layout()
#plt.savefig('images/spectra/lDt.pdf')
plt.show()
