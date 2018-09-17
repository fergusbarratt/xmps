from ncon import ncon
from numpy.random import rand
from numpy import linspace
from numpy import real as re, imag as im
from scipy.linalg import expm
from fMPS import fMPS
from spin import N_body_spins, spins, spinHamiltonians
from mps_examples import bell
import matplotlib.pyplot as plt
Sx, Sy, Sz = spins(0.5)
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

def TDVP(mps_0, h, T):
    """TDVP: Take mps_0, evolve with h according to TDVP
    """
    L = mps_0.L
    h = h.reshape([2, 2]*L)
    def H(h, n, As):
        AL = As[:n-1]
        AR = As[n:]
        cAL = [A.conj() for A in AL]
        cAR = [A.conj() for A in AR]
        if n==1:
            return ncon(cAR+AR+[h], [[1, -1, 3], [2, 3, -4], [1
        if n==L:

    print(H(h, 1, mps_0).shape)


H = spinHamiltonians(0.5, 2).TFIM(10)
T = linspace(0, 1, 100)
mps_0 = bell(1)
psi_0 = mps_0.recombine().reshape(-1)
traj = TDVP(mps_0, H, T)
raise Exception



fig, ax = plt.subplots(3, 1, sharey=True, sharex=True)
ax[0].plot([re(psi_0.conj()@expm(1j*t*H)@Sx1@expm(-1j*t*H)@psi_0) for t in T])
ax[1].plot([re(psi_0.conj()@expm(1j*t*H)@Sy1@expm(-1j*t*H)@psi_0) for t in T])
ax[2].plot([re(psi_0.conj()@expm(1j*t*H)@Sz1@expm(-1j*t*H)@psi_0) for t in T])
ax[0].set_ylim([-1, 1])
ax[1].set_ylim([-1, 1])
ax[2].set_ylim([-1, 1])
plt.show()
