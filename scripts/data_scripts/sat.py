import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from numpy import load, cumsum as cs, arange as ar, expand_dims as ed
from numpy import array, log, save, exp, vectorize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
def av(lys):
    return cs(array(lys), axis=0)/ar(1, len(lys)+1)
Y = []
for D in range(2, 8):
    X = load('data/S_{}.npy'.format(D))
    #plt.plot(av(X), label=str(D))
    Y.append(av(X)[-1])

Ds = array([2, 3, 4, 5, 6, 7])
dat = array(Y)

g_ = interp1d(dat, Ds)

def D(s):
    if s<min(dat):
        return 1
    if s>max(dat):
        return 8
    else:
        return g_(s)

D = vectorize(D)


Ds_ = array([1, 2, 3, 4, 5, 6, 7, 8])
lamb = load('data/exps.npy')
max_l = array(list(map(max, map(abs, lamb))))
ks = array(list(map(sum, map(abs, lamb))))/2

λ = interp1d(Ds_, max_l)
λ_ = interp1d(Ds_, ks)

