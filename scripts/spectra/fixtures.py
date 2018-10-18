import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from fMPS import fMPS
from spin import spins, N_body_spins, n_body
from numpy import load, sum, max, min, real as re
from numpy.linalg import eig
from scipy.linalg import norm
Sx, Sy, Sz = spins(0.5)

psi_0 = load('fixtures/mat8x8.npy')
mps = fMPS().left_from_state(psi_0).right_canonicalise(1)
psi = mps.recombine().reshape(-1)

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
L = 8

bulkH = Sz12@Sz22+Sx22+Sz22
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)]
Hfull = sum([n_body(a, i, len(H), d=2) for i, a in enumerate(H)], axis=0)

l, V = eig(Hfull)
l.sort()
print(min(l), max(l), re(psi.conj().T@Hfull@psi))
