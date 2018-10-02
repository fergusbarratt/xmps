from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.style.use('bmh')

ls = load('data/spectra/maxs.npy')
#f = interp1d([1, 2, 3, 4, 5, 6, 7], ls, kind='cubic')
def g(x, a, b):
    return a*x**b
def f(x): return g(x, *curve_fit(g, [1, 2, 3, 4, 5, 6, 7], ls)[0])

F = load('data/spectra/lambda(D(t)).npy')
plt.plot(F)
plt.show()
