'''store a list of instantaneous lyapunov exponents'''
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI, MPO_XXZ
from numpy.linalg import eigvals
from numpy import abs, kron, eye, array, block, real, imag, min
Sx, Sy, Sz = spins(0.5)
sigma_z = 2*Sz
sigma_y = -2j*Sy

def sympmat(n):
    return kron(sigma_z, eye(n))


Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 3
bulkH =Sx12@Sx22+Sz22@Sz22+Sx12+Sz12
H = [Sx12@Sx22+Sz22@Sz22+Sx12+Sx22+Sx12+Sz22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]
dt = 1e-1
t_fin = 10
T = linspace(0, t_fin, int(t_fin//dt)+1)
t_burn = 0.1
#(100, 10)

mps = fMPS().load('fixtures/product{}.npy'.format(L)).right_canonicalise()
mps = fMPS().random(L, 2, 1).left_canonicalise()
J1, J2, g = mps.jac(H, real_matrix=False)
H = 1j*block([[J1, J2], [J2.conj(), J1]])
Omega = sympmat(H.shape[0]//2)
gs = min(real(eigvals(H)))
print(gs)
H = abs(gs)*eye(H.shape[0]) + H

J = 1j*Omega@H
print(sorted(real(eigvals(H))))
print(sorted(imag(eigvals(J))))
raise Exception

Ds = [2]
for D in Ds:

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'spectra/lyapunovs'


    exps, lys, _ = F.lyapunov(T, D, t_burn=t_burn)

    plt.plot(exps)
    plt.show()
