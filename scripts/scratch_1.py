import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from numpy import load, cumsum as cs, array, arange as ar
import matplotlib.pyplot as plt

def av(lys, dt=1):
    return (1/dt)*cs(array(lys), axis=0)/ar(1, len(lys)+1)

A = load('data/_1/L6_D1_N10000_inst_exps.npy')

plt.plot(A)
plt.savefig('x.pdf')
