from xmps.iMPS import iMPS
from tqdm import tqdm
import numpy as np
from scipy.linalg import norm

def find_ground_state(H, D, ϵ=1e-2, maxiters=5000, tol=1e-3, noisy=False):
    ψ = iMPS().random(2, D).left_canonicalise()
    es = []
    it = tqdm(range(maxiters)) if noisy else range(maxiters)
    for _ in it:
        dA = 1j*ϵ*ψ.dA_dt([H])
        if norm(dA)< tol:
            break
        es.append(ψ.e)
        ψ=(ψ-dA).left_canonicalise()
    return ψ, es
