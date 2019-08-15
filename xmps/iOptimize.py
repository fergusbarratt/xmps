from xmps.iMPS import iMPS
from tqdm import tqdm
import numpy as np
from scipy.linalg import norm

def find_ground_state(H, D, ϵ=1e-1, maxiters=500, tol=1e-4):
    ψ = iMPS().random(2, 2).left_canonicalise()
    es = []
    for _ in range(maxiters):
        dA = 1j*ϵ*ψ.dA_dt([H])
        if norm(dA)< tol:
            break
        es.append(ψ.e)
        ψ=(ψ-dA).left_canonicalise()
    return ψ, es
