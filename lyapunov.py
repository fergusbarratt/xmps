from fMPS import fMPS
from ncon import ncon
from spin import N_body_spins, spins
from numpy import array, linspace, real as re, reshape, sum, swapaxes
from numpy import tensordot as td, squeeze, trace as tr, expand_dims as ed
import numpy as np
from tensor import split_up_3, H as C, C as c, get_null_space
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import sqrtm
from numpy.linalg import cholesky, inv, norm
Sx, Sy, Sz = spins(0.5)
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 3)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 3)
Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 3)

def TDVP(mps, T, H):
    mps.right_canonicalise()
    dt = T[1]-T[0]
    out = [mps]
    e = []
    def A_dot(mps):
        def cache_r(mps):
            rs = [array([[1]])]
            for n in range(len(mps))[::-1]:
                rs.append(sum(mps[n] @ rs[-1] @ C(mps[n]), axis=0))
            return lambda n: rs[::-1][n]

        def cache_l(mps):
            ls = [array([[1]])]
            for n in range(len(mps)):
                ls.append(sum(C(mps[n]) @ ls[-1] @ mps[n], axis=0))
            return lambda n: ls[n]

        l = cache_l(mps)
        r = cache_r(mps)

        def rt(x):
            return sqrtm(x)

        def L(n):
            L_ = C(A[n-1]) @ rt(l(n-1))
            return swapaxes(L_, 0, 1).reshape(L_.shape[1], -1)

        def vL(n):
            """vL: This sometimes produces an axis of zero length i.e. (2, 4, 0)
            when D_{n-1}d = D_n has full rank, the null space is zero dimensional!
            """
            L_ = L(n)
            vL_ = get_null_space(L_)
            return reshape(vL_, (mps.d, int(L_.shape[1]/d), -1))

        def B(x, n):
            return inv(rt(l(n-1))) @ vL(n) @ x @ inv(rt(r(n)))

        psi = squeeze(td(td(A[0], A[1], [-1, 1]), A[2], [-1, 1]))
        Hpsi= td(H, 
                 psi, 
                 [[-3, -2, -1], [0, 1, 2]])

        psiHpsi = td(c(psi), Hpsi, [[0, 1, 2], [0, 1, 2]])

        left1 = td(inv(rt(l(0))), vL(1), [-1, 1])
        right1 = td(td(inv(rt(r(1))), A[1], [-1, 1]), A[2], [-1, 1])
        RHS1 = td(c(left1), td(Hpsi, c(right1), [[-1, 0], [1, 2]]), [-2, 0])

        left2 = td(A[0]@inv(rt(l(1))), vL(2), [-1, 1])
        right2 = inv(rt(r(2)))@A[2]
        RHS2 = td(c(left2), td(Hpsi, c(right2), [-1, 1]), [[0, 2], [0, 1]])

        left3 = td(td(td(A[0], A[1], [-1, 1]), inv(rt(l(2))), [-1, 0]), vL(3), [-1, 1])
        RHS3 = ed(ed(td(c(left3), Hpsi, [[0, 2, 3], [0, 1, 2]]), -1), -1)

        RHS =  [-1j*B(RHS1.reshape(RHS1.shape[1], RHS1.shape[2]), 1),
                -1j*B(RHS2.reshape(RHS2.shape[1], RHS2.shape[2]), 2),
                -1j*B(RHS3.reshape(RHS3.shape[1], RHS3.shape[2]), 3)]

        A_dot = []
        for a, a_dot in zip(A, RHS):
            A_dot.append(a_dot)

        return fMPS(A_dot, 2)

    for _ in T:
        mps = out[-1]
        A = mps.data
        d = mps.d

        k1 = A_dot(mps)*dt
        k2 = A_dot(mps+k1*(1/2))*dt
        k3 = A_dot(mps+k2*(1/2))*dt
        k4 = A_dot(mps+k3)*dt
        
        mps = mps+(k1+k2*2+k3*3+k4)*(1/6)
        out.append(mps.right_canonicalise())

    return out, e

h = 0.1
H = 4*(Sz1@Sz2+Sz2@Sz3) + 2*h*(Sx1+Sx2+Sx3)+0j
#H += Sy1@Sy2
H_ = H.reshape(2, 2, 2, 2, 2, 2)
mps_0 = fMPS().random(3, 2, 20).right_canonicalise()
psi_0 = mps_0.recombine().reshape(-1)

T = linspace(0, 2, 800)
out, _ = TDVP(mps_0, T, H_)
r = array([expm(-1j*t*H)@psi_0 for t in T])
r = array([i/norm(i) for i in r])

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot([re(2*x.E(Sz, 0)) for x in out], c='C0')
ax.plot([re(2*x.E(Sy, 0)) for x in out], c='C1')
ax.plot([re(2*x.E(Sx, 0)) for x in out], c='C2')
ax.plot([2*re(c(r[i, :])@Sz1@r[i, :]) for i in range(len(T))], c='C0')
ax.plot([2*re(c(r[i, :])@Sy1@r[i, :]) for i in range(len(T))], c='C1')
ax.plot([2*re(c(r[i, :])@Sx1@r[i, :]) for i in range(len(T))], c='C2')
ax.set_ylim([-1.1, 1.1])
#ax[1].set_ylim([0, 0.1])
plt.show()

#plt.show()
