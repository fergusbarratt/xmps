import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from numpy import load, save, array
from fMPS import fMPS
from fTDVP import Trajectory
from tdvp.tdvp_fast import MPO_TFI
from spin import N_body_spins
import matplotlib.pyplot as plt
import cProfile

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
L = 30
D = 30
bulkH =Sz12@Sz22+Sx22+Sz22
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

mps = fMPS().random(L, 2, D).right_canonicalise()
T = Trajectory(mps, H=H, W=W)

cProfile.runctx('T.invfree(mps, 0.1)', {'mps':mps, 'T':T}, {}, sort='cumtime')
