import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from numpy import load, save, array
from fMPS import fMPS
from spin import N_body_spins
from time import time
from scipy.linalg import norm
from fTDVP import Trajectory
import matplotlib.pyplot as plt
from numpy import linspace
import cProfile

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
L = 8
D = 8
mps = fMPS().left_from_state(load('fixtures/mat{}x{}.npy'.format(L, L))).right_canonicalise(1).expand(16)
H = [Sz12@Sz22+Sx12+Sx12] +[Sz12@Sz22+Sx22 for _ in range(L-2)]

mps = Trajectory(mps, H).edint(linspace(0, 0.1, 2)).mps

F1, F2 = mps.jac(H, real_matrix=False)
J = mps.jac(H)
print(J.shape)
print(F2)
