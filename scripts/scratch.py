import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from fMPS import fMPS
from fTDVP import Trajectory
from tdvp.tdvp_fast import MPO_TFI
from spin import spins, N_body_spins
from numpy import linspace, array, log, real as re, max
import matplotlib.pyplot as plt

mps = fMPS().load('fixtures/product6.npy').right_canonicalise()
Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 6 
bulkH =Sz12@Sz22+Sx22+Sz22
W = 6*[MPO_TFI(0, 0.5, 0.25, 0.25)]
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)] 

dt = 5e-4
t_fin = 1
T = linspace(0, t_fin, int(t_fin//dt)+1)
T_ = linspace(0, 100, 300)

F = Trajectory(mps, H=H, W=W)
hist = F.invfreeint(T).deserialize()[::100]
ss = []
for x in hist:
    sch = Trajectory(x, H=H, W=W).edint(T_).schmidts()
    λ = array([max([-re(s@log(s)) for s in S]) for S in sch])
    plt.plot(λ)
plt.show()
