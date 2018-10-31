import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from numpy import load, cumsum as cs, arange as ar, expand_dims as ed
from numpy import array, log, save, exp, vectorize, linspace
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

L = 6
D = 8
T = linspace(0, 10, 100)
mps = fMPS().load('fixtures/product{}.npy'.format(L)).right_canonicalise().expand(D)
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

F = Trajectory(mps, H=listH, W=W, fullH=False)
F.run_name = 'spectra/entanglement'
F.invfreeint(T)
sch = array(F.schmidts())
X = array([max([-re(s@log(s)) for s in S]) for S in sch])
#save('data/S_{}'.format(D), λ)

plt.plot(X)
plt.show()
