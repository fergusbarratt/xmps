import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, array
from numpy.linalg import eigvalsh
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

L = 8
ent = Sz1@Sz2
loc = Sx1+Sz1, Sx2+Sz2
listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

e = eigvalsh(fullH)
e_ = [abs(e[i]-e[i+1]) for i in range(len(e)-1)]
plt.hist(e_, bins=30)
plt.show()
