import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,os.path.dirname(parentdir))

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, array, concatenate as ct, stack as st, log, exp
from numpy import ceil, real as re, cumsum as cs, arange as ar, array
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 6 
bulkH =Sz12@Sz22+Sx22
H = [Sz12@Sz22+Sx12+Sx22]+[bulkH for _ in range(L-2)]

dt = 0.01
t_fin = 100
D = 8
T = linspace(0, t_fin, int(t_fin//dt)+1)
psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))

mps = fMPS().left_from_state(psi_0).left_canonicalise(1).expand(D)

F = Trajectory(mps, H=H)
F.run_name = 'spectra/entanglement'
F.edint(T)
sch = array(F.schmidts())
λ = array([exp(max([-re(s@log(s)) for s in S])) for S in sch])
#save('data/spectra/Dt', λ)

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(T[1:], λ)
ax.set_title('$D(t)$', loc='right')

fig.tight_layout()
fig.savefig('images/spectra/Dt.pdf')
plt.show()
