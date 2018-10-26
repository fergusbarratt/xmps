'''store a list of instantaneous lyapunov exponents'''
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 8 
bulkH =Sz12@Sz22+Sx12+Sx22
H_i = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]
H = [H_i[0]+Sz12]+[H_i[i]+Sz12+Sz22 for i in range(1, L-2)]+[H_i[-1]+Sz22]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

dt = 5e-2
t_fin = 10 
T = linspace(0, t_fin, int(t_fin//dt)+1)
t_burn = 1
load_basis = True
Q = None
#(100, 10)

if L<10:
    psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))
    mps = fMPS().left_from_state(psi_0).right_canonicalise(1)
else:
    mps = fMPS().load('fixtures/product{}.npy'.format(L))


Ds = [2]
for D in Ds:
    if load_basis and 2==D:
        Q = load('data/bases/spectra/lyapunovs_L8_D2_N10000_basis.npy')
        mps = fMPS().load('data/bases/spectra/lyapunovs_L8_D2_N10000_state.npy')
    if load_basis and 3==D:
        Q = load('data/bases/spectra/lyapunovs_L8_D3_N20000_basis.npy')
        mps = fMPS().load('data/bases/spectra/lyapunovs_L8_D3_N20000_state.npy')
    if load_basis and 4==D:
        Q = load('data/bases/spectra/lyapunovs_L8_D4_N20000_basis.npy')
        mps = fMPS().load('data/bases/spectra/lyapunovs_L8_D4_N20000_state.npy')
    if load_basis and 5==D:
        Q = load('data/bases/spectra/lyapunovs_L8_D5_N20000_basis.npy')
        mps = fMPS().load('data/bases/spectra/lyapunovs_L8_D5_N20000_state.npy')

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'spectra/lyapunovs'
    exps, lys = F.lyapunov(T, D, t_burn=t_burn, basis=Q)
    F.save(exps=True)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(lys)
    ax[1].plot(exps)
    plt.show()