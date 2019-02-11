import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from numpy import load, cumsum as cs, array, arange as ar, expand_dims as ed
import matplotlib.pyplot as plt

def av(lys, dt=1):
    return (1/dt)*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), -1)

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for D in [2, 3]:
    A = load('data/spectra/lyapunovs_{}/L4_D{}_N10000_inst_exps.npy'.format(D, D))
    B = load('data/spectra/lyapunovs_{}/L4_D{}_N10000_inst_exps_.npy'.format(D, D))

    ax[D-1][0].plot(av(A[10:], 1e-3))
    ax[D-1][1].plot(av(B[10:], 1e-3))
    ax[D-1][2].plot(av((A[10:]+B[10:])/2, 1e-3))

plt.savefig('x.pdf')
