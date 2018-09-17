from fMPS import fMPS
from spin import N_body_spins, spins, spinHamiltonians
from ncon import ncon
from numpy import array, linspace, real as re, reshape, sum, swapaxes
from numpy import tensordot as td, trace as tr, expand_dims as ed
from numpy import tensordot as td, squeeze as sq, swapaxes as sw
from numpy import transpose as tra, allclose
from mps_examples import bell
from numpy import linspace, prod, ones_like
from numpy.random import rand
from tensor import C as c, H as he
import matplotlib.pyplot as plt
from functools import reduce
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from scipy.linalg import expm, qr, rq
from numpy.linalg import qr as qr_n, norm
from expokitpy import zgexpv, zhexpv
import scipy as sp
from copy import deepcopy
Sx, Sy, Sz = spins(0.5)
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

def lanczos_expm(A, v, t, norm_est=1., m=5, tol=0., trace=False, A_is_Herm=False):
    ideg = 6
    #Override expokit default precision to match scipy sparse eigs more closely.
    if tol == 0:
        tol = sp.finfo(sp.complex128).eps * 2 #cannot take eps, as expokit changes this to sqrt(eps)!

    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)

    m = min(xn - 1, m)

    nwsp = max(10, xn * (m + 2) + 5 * (m + 2)**2 + ideg + 1)
    wsp = sp.zeros((nwsp,), dtype=A.dtype)

    niwsp = max(7, m + 2)
    iwsp = sp.zeros((niwsp,), dtype=sp.int32)

    iflag = sp.zeros((1,), dtype=sp.int32)
    itrace = sp.array([int(trace)])

    output_vec,tol0,iflag0 = zgexpv(m,t,v,tol,norm_est,wsp,iwsp,A.matvec,0)

    if iflag0 == 1:
        print("Max steps reached!")
    elif iflag0 == 2:
        print("Tolerance too high!")
    elif iflag0 < 0:
        print("Bad arguments!")
    elif iflag0 > 0:
        print("Unknown error!")

    return output_vec

def TDVP(mps, T, H_):
    dt = T[1]-T[0]
    out = []
    e = []
    L = mps.L
    d = mps.d
        
    # sweep
    def sweep(mps):
        mps = fMPS(mps.data, 2)
        def H(n, AL, AR):
            h = H_.reshape([2, 2]*L)
            #AL = mps.data[:n-1]
            #AR = mps.data[n:]
            if n==1:
                AR = sw(sq(reduce(lambda x, y: td(x, y, [-1, -2]), AR)), 0, 1) #(2, 2) (aux, spin)
                ALH = ed(h, 0) #(1, 2?, 2_, 2, 2) @ (2, 2_(sp)), [[2], [1]] (indexing is not backwards on the bottom)
                ALHAR = td(ALH, c(AR), [range(2, L-n+2), range(1, L-n+1)]) #(1, 2(sp), 2(sp), 2(sp), 2(aux, down))
                ALALHAR = ed(ALHAR, 0) #(1, 1, 2, 2, 2_(sp), 2(aux)) @ (2, 2_(sp)), [[4], [1]]
                Hn = td(ALALHAR, AR, [range(4, L-n+4), range(1, L-n+1)]) #(1(aux, down), 1(aux, up), 2(sp, down), 2(sp, up), 2(aux, down), 2(aux, up))
                AC = mps.data[n-1]
                #print(td(td(c(AC), Hn, [[0, 1, 2], [2, 0, 4]]), AC, [[1, 0, 2], [0, 1, 2]]))
                #print(c(mps.recombine().reshape(-1))@H_@mps.recombine().reshape(-1))
            elif n==L:
                AL = sq(reduce(lambda x, y: td(x, y, [-1, -2]), AL)) #(2, 2) (spin, aux)
                #(2_, 2, 2, 2), (2_, 2) [[0], [0]]
                ALH = td(c(AL), h, [range(n-1), range(n-1)]) #(2(aux), 2(sp), 2(sp), 2(sp))
                ALHAR = ed(ALH, -1) #(2, 2, 2, 2, 1)
                #(2_, 2), (2, 2, 2_, 2, 1), [[0], [2]]
                ALALHAR = sw(td(AL, ALHAR, [range(n-1), range(2, n+1)]), 0, 1) # (2(aux, down), 2(aux, up), 2(sp, down), 2(sp, up), 1(aux), 1(aux))
                Hn = ed(ALALHAR, -1)
                AC = mps.data[n-1]
                #print(td(td(c(AC), Hn, [[0, 1, 2], [2, 0, 4]]), AC, [[1, 0, 2], [0, 1, 2]]))
                #print(c(mps.recombine().reshape(-1))@H_@mps.recombine().reshape(-1))
            else:
                AL = sq(reduce(lambda x, y: td(x, y, [-1, -2]), AL))
                AR = sw(sq(reduce(lambda x, y: td(x, y, [-1, -2]), AR)), 0, 1)
                ALH = td(c(AL), h, [range(n-1), range(n-1)])
                ALHAR = td(ALH, c(AR), [range(2, L-n+2), range(1, L-n+1)])
                ALALHAR = td(AL, ALHAR, [range(n-1), range(3, n+2)])
                Hn = td(ALALHAR, AR, [range(4, L-n+4), range(1, L-n+1)])

            return tra(Hn, [0, 2, 4, 1, 3, 5])

        def K(n, AL, AL_c, AR):
            #td(H(n).shape, mps.data[n-1].shape), [[3, 1], [0, 1]]) (1, 2, 2, 1_, 2_, 2), (2_, 1_, 2): 
            #(2_, 1_, 2), (1_(aux, dl), 2_(sp, d), 2(aux, dr), 2(aux, ur), 2(aux, ul))
            #(2(aux, dl), 2(aux, dr), 2(aux, ur), 2(aux, ul))
            return ncon([c(AL_c), H(n, AL[:-1], AR), AL_c], [[1, 2, -1], [2, 1, -2, 4, 3 ,-4], [3, 4, -3]])

        n = 1
        A = mps.data[0]
        C_new = array([[1]])
        AC_new = (expm(-1j/2*dt*(H(n, [], [mps.data[1]]).reshape((4, 4))))@((C_new@A).reshape(-1))).reshape(A.shape)
        AL, C = qr(AC_new.reshape((prod(A.shape[:2]), -1)), mode='economic')
        mps.data[0] = AL.reshape(A.shape)
        C_new = (expm(1j/2*dt*K(n, [], mps.data[0], [mps.data[1]]).reshape((4, 4)))@(C.reshape(-1))).reshape(C.shape)

        n = 2
        A = mps.data[-1]
        AC_new = (expm(-1j/2*dt*H(n, [mps.data[0]], []).reshape((4, 4)))@(C_new@A).reshape(-1)).reshape(A.shape)
        AC, C = qr(AC_new.reshape((prod(AC_new.shape[:2]), -1)), mode='economic')
        mps.data[-1] = AC.reshape(A.shape)
        mps.right_canonicalise()

        #n = 1
        #A = mps.data[-1]
        #C, AR = rq(sw(A, 0, 1).reshape(A.shape[1], -1), mode='economic')
        #mps.data[-1] = sw(AR.reshape((A.shape[1], A.shape[0], -1)), 0, 1)
        #C_new = (expm(1j/2*dt*K(n, [], mps.data[0], [mps.data[1]]).reshape((4, 4)))@(C.reshape(-1))).reshape(C.shape)

        #A = mps.data[0]
        #AC_new = (expm(-1j/2*dt*H(n, [], [mps.data[1]]).reshape((4, 4)))@((A@C_new).reshape(-1))).reshape(A.shape)
        #C, AR = rq(sw(AC_new, 0, 1).reshape(A.shape[1], -1), mode='economic')
        #mps.data[0] = sw(AR.reshape((A.shape[1], A.shape[0], -1)), 0, 1)

        return mps
        
    from tests import is_right_canonical, is_left_canonical
    for _ in T:
        out.append([ncon([c(mps.data[0]), Sx, mps.data[0]], [[1, 3, 4], [1, 2], [2, 3, 4]])])
        mps = sweep(mps)

    return array(out)

mps_0 = fMPS().random(2, 2, 20).right_canonicalise()
psi_0 = mps_0.recombine().reshape(-1)
T1 = linspace(0, 1, 1000)
T2 = linspace(0, 1, 1000)
H = Sx1@Sx2+Sy1@Sy2+Sz1@Sz2
Tra = TDVP(mps_0, T1, H)
fig, ax = plt.subplots(1, 1)
ax.plot([c(psi_0)@expm(1j*t*H)@Sx1@expm(-1j*t*H)@psi_0 for t in T2])
ax.plot(Tra[:, 0])
#ax.set_ylim([-1, 1])
#ax.set_ylim([-1, 1])
plt.show()
