from pymps.iMPS import iMPS
import matplotlib.pyplot as plt
from pymps.spin import spins, N_body_spins
from tqdm import tqdm 

Sx, Sy, Sz = spins(0.5) 
Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

N, dt = 100, -0.1j
d, D = 2, 30 
λ = 1
A = iMPS().random(d, D).left_canonicalise()
def H(λ): return [Sz12@Sz22+λ*(Sx12+Sx22)]
H = H(λ)

es = []
for _ in tqdm(range(N)):
    dA = A.dA_dt(H)*dt
    es.append(A.e)
    A = (A+dA).mixed()[0]
plt.plot(es)
plt.show()
