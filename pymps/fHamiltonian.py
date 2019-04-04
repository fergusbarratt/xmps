import unittest
from numpy.random import randint, rand
from numpy import zeros, identity, array, swapaxes, dot, tensordot
from numpy import transpose, diag, prod, kron, allclose
from numpy.linalg import svd
from tensor import H, C
from qmb import sigmap, sigmam, sigmaz
from fMPS import fMPS
from scipy.sparse.linalg import eigsh as lanczos, LinearOperator
import matplotlib.pyplot as plt
from tests import is_left_canonical, is_right_canonical

class effH(object):
    def __init__(self, W, A, k, L, R):
        self.W = W
        self.A = A
        if k is not None and L is not None and R is not None:
            self.k = k-1
            self.L = L
            self.R = R
            self.shape = (self.W[self.k].shape[0]*prod(self.A[self.k].shape[1:]),
                        self.W[self.k].shape[1]*prod(self.A[self.k].shape[1:]))

    def env_right(self, L, k):
        W, A = self.W, self.A
        return transpose(tensordot(tensordot(dot(H(A[k-1]), L),
                                                W[k-1], [[0, 2], [0, 2]]),
                                    A[k-1], [[2, 1], [0, 1]]),
                            [1, 0, 2])

    def env_left(self, R, k):
        W, A = self.W, self.A
        return transpose(tensordot(A[k-1],
                                    tensordot(W[k-1],
                                                dot(R, H(A[k-1])),
                                                [[0, 3], [2, 0]]),
                                    [[0, 2], [0, 2]]),
                            [1, 0, 2])

    def get_left_env(self, k):
        N = len(self.A)
        L1 = array([[[1.]]])
        Ls = [L1]

        for k in range(1, k):
            Ls.append(self.env_right(Ls[-1], k))

        return lambda n: Ls[n-1]

    def get_right_env(self, k_s=0):
        N = len(self.A)
        RL_1 = array([[[1.]]])
        Rs = [RL_1]

        for k in range(k_s+1, N+1)[::-1]:
            Rs.append(self.env_left(Rs[-1], k))

        return lambda n: Rs[::-1][n-1]

    def get_envs(self, k):
        self.k = k
        self.L = self.get_left_env(k+1)(0)
        self.R = self.get_right_env(k+1)(1)
        self.shape = (self.W[k].shape[0]*prod(self.A[k].shape[1:]),
                      self.W[k].shape[1]*prod(self.A[k].shape[1:]))
        return self

    def mv(self, A):
        A = A.reshape(self.A[self.k].shape)
        W = self.W[self.k]
        L = self.L
        R = self.R
        RA = tensordot(R, A, [-1, -1])
        WRA = tensordot(W, RA, [[1, 3], [2, 0]])
        LWRA = tensordot(L, WRA, [[0, 2], [1, 3]])
        return LWRA

    def asmatrix(self):
        W = self.W[self.k]
        L = self.L
        R = self.R
        WR = tensordot(W, R, [3, 0]) #(i, i', k, b', b)
        LWR = transpose(tensordot(L, WR, [0, 2]),
                        [2, 0, 4, 3, 1, 5])
        return LWR.reshape(prod(LWR.shape[:3]), prod(LWR.shape[3:]))

    def aslinearoperator(self):
        return LinearOperator(shape=self.shape,
                              matvec=self.mv)

    def optimize(self):
        v0 = self.A[self.k].reshape(self.shape[-1])
        e, v = lanczos(self.aslinearoperator(), k=1, which='SA', v0=v0)
        return e, v.reshape(self.A[self.k].shape)

class fMPO(object):
    """finite MPO"""
    def __init__(self, data=None):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def anisotropic_heisenberg(self, L, h, J, Jz):
        """set internal MPO to anisotropic_heisenberg MPO

        :param L: length of chain
        :param h: h sum Szi
        :param J: J sum S+iS-i+1 + conj
        :param Jz: Jz sum SziSzi+1
        """
        self.d = 2
        self.L = L

        O = zeros((2, 2))
        I = identity(2)
        sm = sigmam().full()
        sp = sigmap().full()
        sz = sigmaz().full()

        first = array([[-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
        bulk = array([[I, O, O, O, O],
                      [sp, O, O, O, O],
                      [sm, O, O, O, O],
                      [sz, O, O, O, O],
                      [-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
        last = array([[I], [sp], [sm], [sz], [-h*sz]])
        fMPO = [first]+[bulk]*(L-2)+[last]

        for i, mats in enumerate(fMPO):
            fMPO[i] = swapaxes(swapaxes(mats, 0, 2), 1, 3)

        self.data = fMPO
        return self

    def DMRG(self, n_iters, D, mps=None):
        if mps == None:
            mps = fMPS().random(self.L, self.d, D).right_canonicalise()
        W = self.data
        A = mps.data
        N = self.L

        def gauge_right(A, k):
            Ak = transpose(A[k-1], [1, 0, 2]).reshape(A[k-1].shape[0]*A[k-1].shape[1], -1)
            U, S, V = svd(Ak, full_matrices=False)
            Ak = transpose(U.reshape(A[k-1].shape[1], A[k-1].shape[0], A[k-1].shape[2]),
                           [1, 0, 2])
            Akp1 = diag(S) @ V @ A[k]
            return Ak, Akp1

        def gauge_left(A, k):
            Ak = transpose(A[k-1], [1, 0, 2]).reshape(A[k-1].shape[1], -1)
            U, S, V = svd(Ak, full_matrices=False)
            Ak = transpose(V.reshape(A[k-1].shape[1], A[k-1].shape[0], A[k-1].shape[2]),
                           [1, 0, 2])
            Akm1 = A[k-2] @ U @ diag(S)
            return Akm1, Ak

        def env_right(W, A, L, k):
            return transpose(tensordot(tensordot(dot(H(A[k-1]), L),
                                                 W[k-1], [[0, 2], [0, 2]]),
                                       A[k-1], [[2, 1], [0, 1]]),
                             [1, 0, 2])

        def env_left(W, A, R, k):
            return transpose(tensordot(A[k-1],
                                       tensordot(W[k-1],
                                                 dot(R, H(A[k-1])),
                                                 [[0, 3], [2, 0]]),
                                       [[0, 2], [0, 2]]),
                             [1, 0, 2])

        def get_all_left_envs():
            L1 = array([[[1.]]])
            Ls = [L1]

            for k in range(1, N):
                Ls.append(env_right(W, A, Ls[-1], k))

            return lambda n: Ls[n-1]

        def get_all_right_envs():
            RL_1 = array([[[1.]]])
            Rs = [RL_1]

            for k in range(1, N+1)[::-1]:
                Rs.append(env_left(W, A, Rs[-1], k))

            return lambda n: Rs[::-1][n-1]

        def optimize_at_site(W, A, k, L, R):
            return effH(W, A, k, L, R).optimize()

        def right_sweep(A):
            L = array([[[1.]]])
            es = []
            for k in range(1, N):
                R = get_all_right_envs()
                if not is_left_canonical(A[:k-1]) and is_right_canonical(A[k:]):
                    raise Exception
                e, A[k-1] = optimize_at_site(W, A, k, L, R(k+1))
                es.append(e)
                A[k-1], A[k] = gauge_right(A, k)
                L = env_right(W, A, L, k)
            return es

        def left_sweep(A):
            R = array([[[1.]]])
            es = []
            for k in range(2, N+1)[::-1]:
                L = get_all_left_envs()
                if not is_left_canonical(A[:k-1]) and is_right_canonical(A[k:]):
                    raise Exception
                e, A[k-1] = optimize_at_site(W, A, k, L(k), R)
                es.append(e)
                A[k-2], A[k-1] = gauge_left(A, k)
                R = env_left(W, A, R, k)
            return es

        es = []
        for _ in range(n_iters):
            e_r = right_sweep(A)
            print(e_r)
            e_l = left_sweep(A)
            print(e_l)
            es = es+e_r+e_l

        return es, fMPS(A, self.d)

class TesteffH(unittest.TestCase):
    def setUp(self):
        """setUp"""
        self.L = 2 
        self.h = 0
        self.J = 1
        self.Jz = 1

        self.W = fMPO().anisotropic_heisenberg(self.L, self.h, self.J, self.Jz).data
        self.A = fMPS().random(self.L, 2, 5).data

    def test_matrix_linearoperator(self):
        h = effH(self.W, self.A, None, None, None).get_envs(1)
        x = rand(h.shape[1])
        print(h.asmatrix() @ x, h.aslinearoperator() @ x, sep='\n\n')
        self.assertTrue(allclose(h.asmatrix() @ x, h.aslinearoperator() @ x))

class TestfMPO(unittest.TestCase):
    """TestfMPO"""

    def setUp(self):
        """setUp"""
        self.L = 2
        self.h = 0
        self.J = 1
        self.Jz = 1

        self.s182 = fMPO().anisotropic_heisenberg(self.L, self.h, self.J, self.Jz)

if __name__ == '__main__':
    unittest.main(verbosity=1)
