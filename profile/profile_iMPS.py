from pymps.iMPS import iMPS
from pymps.spin import spins, N_body_spins
import cProfile
Sx, Sy, Sz = spins(0.5) 
Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

mps = iMPS().load('testing/fixtures/iMPS2x50.npy').left_canonicalise()
def H(λ): return [Sz12@Sz22+λ*(Sx12+Sx22)]
λ = 1
mps.dA_dt(H(λ))

#cProfile.runctx('mps.dA_dt(H(λ))', {'mps':mps, 'H':H, 'λ':λ}, {}, sort='cumtime')
