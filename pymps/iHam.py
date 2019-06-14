'''Module containing representations of TI infinite hamiltonians'''
from .iMPS import iMPS
from .spin import spins, N_body_spins
from numpy import copy, transpose, prod
from numpy.linalg import cholesky as ch
from .tensor import C as c, embed, unitary_extension, direct_sum
from .ncon import ncon
from numpy import zeros_like as zl, array, zeros as z, block , eye
from numpy.linalg import inv
from scipy.linalg import norm, expm

d = 2
s = (d-1)/2
Sx, Sy, Sz = spins(s)
Sx12, Sy12, Sz12 = N_body_spins(s, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(s, 2, 2)
def heis(Δ=1): return [Sx12@Sx22+Sy12@Sy22+Δ*Sz12@Sz12]
def isin(λ=1/2): return [-Sz12@Sz22+λ*(Sx12+Sx22)]

def ungauge(A, conj=False):
    def co(x): return x if not conj else c(x)
    tens = A.data[0]
    k = len(tens.shape[3:])
    links = [1, 2, -2]+list(range(-3, -3-k, -1))

    l, r, vL = A.l, A.r, A.vL
    return ncon([ch(l)@co(vL),
                ncon([tens, ch(r)], [[-1, -2, 1], [1, -3]])],
                [[1, 2, -1], links])

def optimise(H, D, method='euler', max_iters=500, tol=1e-5):
    if method == 'euler':
        A = iMPS().random(d, D).canonicalise('m')
        δt = -1e-1j
        e_ = A.energy(H)
        for _ in range(max_iters):
            A = (A+δt*A.dA_dt(H)).canonicalise('m')

            e = A.energy(H)
            ϵ = abs(e-e_)
            if ϵ < tol:
                break

            e_ = copy(e)
        return A, A.energy(H)
    elif method == 'unitary':
        A = iMPS().random(d, D).canonicalise('r')
        As = [transpose(a, [1, 0, 2]) for a in A.data]
        isometries = [a.reshape(-1, prod(a.shape[1:])) for a in As]
        Us = [unitary_extension(Q, 4) for Q in isometries]
        U = Us[0]
        # U = (A, vL)

        dA = A.dA_dt(H)
        l, r, vL = dA.l, dA.r, dA.vL
        # assume right canonical
        Λl = ch(l)
        Λr = ch(r)
        dx = ungauge(dA)

        ϵ = 1e-1
        dA *= ϵ
        dx *= ϵ
        O1 = z([dx.shape[0]]*2)
        O2 = z([dx.shape[1]]*2)

        X = block([[O1,                 dx.conj().T],
                   [dx,          O2]])

        L = direct_sum(eye(D), inv(Λl))
        R = direct_sum(eye(D), inv(Λr))
        Q = expm(-1j*X)
        U = L@U@Q@R
        # (A, vL) -> (A+inv(l)-vL-dx-inv(r)-, v') (v' orthogonal)
        print(U@U.conj().T)

        raise Exception

