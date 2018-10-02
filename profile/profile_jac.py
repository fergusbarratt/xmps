import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from numpy import load
from fMPS import fMPS
from spin import N_body_spins
import cProfile

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
L = 8 
mps = fMPS().left_from_state(load('fixtures/mat8x8.npy')).left_canonicalise(3)
H = [Sz12@Sz22+Sx12] +[Sz12@Sz22+Sx12+Sx22 for _ in range(L-3)]+[Sz12@Sz22+Sx22]
#mps.jac(H)
cProfile.runctx('mps.jac(H)', {'mps':mps, 'H':H}, {}, sort='cumtime')
