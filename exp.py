from numpy import zeros, diag, real as re
from numpy.linalg import norm
from scipy.linalg import expm
from numpy.random import randn
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import expm
from tensor import C as c
from numba import jit

def expm_lanczos(H, v, dt, k):
    vs = 1j*zeros((v.shape[0],k))
    v /= norm(v)
    vs[:,0] = v

    ɑ, β = 1j*zeros(k), 1j*zeros(k)

    w = H@v

    ɑ[0] = c(w)@v  
    w = w -  ɑ[0]*vs[:, 0]
    β[1] = norm(w)
    vs[:,1] = w/β[1]

    for j in range(1,k-1):
            w =   H@vs[:, j] - β[j]*vs[:, j-1]
            ɑ[j] = re(c(w)@vs[:,j])
            w = w -  ɑ[j] * vs[:,j]
            β[j+1] = norm(w) 
            vs[:,j+1] = w/β [j+1]

    w = H@vs[:, k-1] - β[k-1]*vs[:, k-2]
    ɑ[k-1] = re(c(w)@vs[:, k-1])

    T = diag(ɑ, 0) + diag(β[1:k],1) + diag(β[1:k],-1) 

    n = 1j*zeros(k)
    n[0] = 1.

    return vs@(expm(dt*T)@n)
