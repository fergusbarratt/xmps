import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, abs, array
from numpy.linalg import eig
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI
Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 2 
bulkH =Sz12@Sz22+Sx22+Sz22
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]
fullH = sum([n_body(a, i, len(H), d=2) for i, a in enumerate(H)], axis=0)
mps = fMPS().random(L, 2, 4).left_canonicalise()
D, U = eig(fullH)
print(D)
eigenstates = [fMPS().left_from_state(u.reshape([2]*L)).right_canonicalise() for u in U.T]
overlaps = []
for _ in range(20):
    overlaps.append([m.overlap(mps).imag for m in eigenstates])
    mps = fMPS().left_from_state((expm(-1j*fullH*0.1)@mps.recombine().reshape(-1)).reshape([2]*L))
plt.plot(array(overlaps))
plt.show()
