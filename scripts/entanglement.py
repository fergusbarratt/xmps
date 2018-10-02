import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, array, concatenate as ct, stack as st, log, exp
from numpy import ceil, real as re, cumsum as cs, arange as ar
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI
from scipy.interpolate import interp1d

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 8 
bulkH =Sz12@Sz22+Sx22
H_i = [Sz12@Sz22+Sx12+Sx22] + [bulkH for _ in range(L-2)]
H = [H_i[0]+Sz12+Sz22]+[H_i[i]+Sz22 for i in range(1, L-1)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

dt = 5e-3
t_fin = 30 
D = 16 
T = linspace(0, t_fin, int(t_fin//dt)+1)
psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))

mps = fMPS().left_from_state(psi_0).left_canonicalise(1).expand(D)

ls = load('data/spectra/maxs.npy')
f = interp1d([1, 2, 3, 4, 5, 6, 7], ls)

F = Trajectory(mps, H=H, W=W)
F.run_name = 'spectra/entanglement'
F.invfreeint(T)
sch = F.schmidts()
evs = F.mps_evs((Sx,), 0)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(T, evs)
λ = array([f(exp(max([-re(s[i//2]@log(s[i//2])) for i in range(len(sch[0]))]))) for s in sch])
save('data/spectra/lambda(D(t))', λ)
ax[1].plot(T[1:], λ)
ax[1].set_title('$\lambda(D(t))$', loc='right')
fig.tight_layout()

plt.show()
