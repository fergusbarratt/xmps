import unittest
from numpy.random import randn
from scipy.sparse.linalg import expm_multiply as expmul, LinearOperator
from scipy.sparse.linalg import eigsh as lanczos
from fMPS import fMPS
from MPO import fMPO
from numpy import identity, ravel, array, transpose, tensordot, dot, prod
from tensor import T, H
from scipy.sparse.linalg import expokit

class fastH(object):

    def __init__(self, mpo, mps, oc):
        self.mpo = mpo
        self.mps = mps
        self.oc = oc
        self.shape = (self.mpo[oc].shape[0]*prod(self.mps[oc].shape[1:]),
                      self.mpo[oc].shape[1]*prod(self.mps[oc].shape[1:]))

    def update_mps(self, mps):
        self.mps = mps

    def left(self, k=None):
        """return a function L(n)"""
        A = self.mps
        W = self.mpo
        N = len(self.mps)
        if k is None:
            k = N
        def next_L(L_, l_):
            return transpose(tensordot(tensordot(dot(H(A[l_-1]), L_),
                                                 W[l_-1], [[0, 2], [0, 2]]),
                                       A[l_-1], [[2, 1], [0, 1]]),
                             [1, 0, 2])

        def get_L(k):
            L1 = array([[[1.]]])
            Ls = [L1]

            for l_ in range(1, k):
                Ls.append(next_L(Ls[-1], l_))

            return lambda n: Ls[n-1]

        return get_L(k)

    def right(self, k_s=0):
        """return a function R(n)"""
        A = self.mps
        W = self.mpo
        N = len(self.mps)
        def next_R(R_, l_):
            return transpose(tensordot(A[l_-1],
                                       tensordot(W[l_-1],
                                                 dot(R_, H(A[l_-1])),
                                                 [[0, 3], [2, 0]]),
                                       [[0, 2], [0, 2]]),
                             [1, 0, 2])

        def get_R(k_s):
            RL_1 = array([[[1.]]])
            Rs = [RL_1]

            for l_ in range(k_s+1, N+1)[::-1]:
                Rs.append(next_R(Rs[-1], l_))

            return lambda n: ([None]*k_s + Rs[::-1])[n-1]

        return get_R(k_s)

    def mv(self, A):
        A_old = self.mps[self.oc]
        A = A.reshape(A_old.shape)
        W = self.mpo[self.oc]
        L = self.left(self.oc+1)(self.oc+1)
        R = self.right(self.oc+1)(self.oc+2)
        RA = tensordot(R, A, [-1, -1])
        WRA = tensordot(W, RA, [[1, 3], [2, 0]])
        LWRA = tensordot(L, WRA, [[0, 2], [1, 3]])
        return LWRA

    def mvr(self, A):
        return self.mv(A)

    def aslinearoperator(self):
        return LinearOperator(shape=self.shape,
                              matvec=self.mv,
                              rmatvec=self.mvr)



class TestfastH(unittest.TestCase):
    def setUp(self):
        L, d, D = 10, 2, 5
        h, J, Jz = 0.1, 0.2, 0.3
        oc = 0
        self.mps = fMPS().random(L, d, D).mixed_canonicalise(oc)
        self.mpo = fMPO().anisotropic_heisenberg(L, h, J, Jz)

    def test_fastH(self):
        h = fastH(self.mpo, self.mps, self.mps.oc).aslinearoperator()
        print(h.shape)
        print(lanczos(h, k=1)[0])

def invfree(mps, h, dt, n, ops=[]):
    for i, A in enumerate(mps):
        h_i = fastH(h, mps, i).aslinearoperator()
        print((h_i @ ravel(A)).shape)
        A_new = expmul(-i*h_i*(dt/2.), ravel(A)).reshape(A.shape)
        print(A_new.shape)
    print(mps[0].shape)
    print(h[0].shape)


class TestTDVP(unittest.TestCase):
    def test_invfree_runs(self):
        d, D, L = 2, 5, 5

        h, J, Jz = 0.2, 0.3, 0.4

        dt, n = 0.1, 5

        mps = fMPS().random(L, d, D).right_canonicalise()
        h = fMPO().anisotropic_heisenberg(L, h, J, Jz)
        ops = [identity(2)]
        invfree(mps, h, dt, n, ops)

if __name__ == '__main__':
    unittest.main(verbosity=2)
