import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from numpy import array
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 4 
bulkH =Sz12@Sz22+Sx12+Sx22
H_i = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]
H = [H_i[0]+Sz12]+[H_i[i]+Sz12+Sz22 for i in range(1, L-2)]+[H_i[-1]+Sz22]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

dt = 5e-3
t_fin = 1
T = linspace(0, t_fin, int(t_fin//dt)+1)

if L<10:
    psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))
    mps = fMPS().left_from_state(psi_0).right_canonicalise(1)
else:
    mps = fMPS().load('fixtures/product{}.npy'.format(L))

Ds = [2]
fig, ax = plt.subplots(3, 1, sharex=True)
for D in Ds:
    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'spectra/lyapunovs'
    exps, lys = F.lyapunov(T, D, m=1, t_burn=0.1)
    ax[0].plot(F.mps_history[F.burn_len:])
    ax[1].plot(lys)
    ax[2].plot(array(F.vs))
    plt.show()
    #F.save(exps=True)
