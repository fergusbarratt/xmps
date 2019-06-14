
from .fMPS import fMPS
from .spin import spins, N_body_spins
from numpy import copy, sqrt
d = 3
s = (d-1)/2
Sx, Sy, Sz = spins(s)
Sx12, Sy12, Sz12 = N_body_spins(s, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(s, 2, 2)
def heis(L, Δ=1): return (L-1)*[Sx12@Sx22+Sy12@Sy22+Δ*Sz12@Sz12]
def isin(L, λ=1/2): return [-Sz12@Sz22+λ*Sx12]+(L-2)*[-Sz12@Sz22+λ*(Sx12+Sx22)]

def optimise(H, L, D, method='euler', max_iters=500, tol=1e-5):
    if method == 'euler':
        A = fMPS().random(L, d, D).left_canonicalise()
        δt = -1e-1j
        e_ = A.energy(H)
        for _ in range(max_iters):
            A = (A+δt*A.dA_dt(H)).left_canonicalise()

            e = A.energy(H)
            ϵ = abs(e-e_)
            if ϵ < tol:
                break

            e_ = copy(e)
        return A, A.energy(H)/L
