import unittest
from xmps.fMPS import fMPS
from xmps.fTDVP import Trajectory
from xmps.ncon import ncon
from numpy.random import randint, randn
from numpy import kron, swapaxes as sw, eye, transpose as tra, sqrt
from numpy import allclose, linspace, load, real as re, sum
from numpy import tensordot, trace, dot
from scipy.linalg import norm, null_space as null, inv, cholesky as ch
from scipy.linalg import block_diag as bd
from xmps.tensor import H as cT, C as c
from xmps.spin import N_body_spins, spins
import matplotlib.pyplot as plt
Sx, Sy, Sz = spins(0.5)

def fs(X):
    return X.reshape(int(sqrt(X.shape[0])), int(sqrt(X.shape[0])), *X.shape[1:]
                     ).transpose(1, 0, *list(range(2, len(X.shape)+1))
                     ).reshape(X.shape) 

class fTFD(fMPS):
    def __init__(self, data=None, d=None, D=None):
        """__init__"""
        super().__init__(data, d, D)

    def __add__(self, other):
        """__add__: This is not how to add two TFD: it's itemwise addition.
                    A hack for time evolution.

        :param other: TFD with arrays to add
        """
        return fTFD([a+b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        """__sub: This is not how to subtract two TFD: it's itemwise addition.
                    A hack for time evolution.

        :param other: TFD with arrays to subtract
        """
        return fTFD([a-b for a, b in zip(self.data, other.data)])

    def __mul__(self, other):
        """__mul__: This is not multiplying an TFD by a scalar: it's itemwise:
                    Hack for time evolution.
                    Multiplication by other**L

        :param other: scalar to multiply
        """
        return fTFD([other*a for a in self.data], self.d)

    def __rmul__(self, other):
        """__rmul__: right scalar multiplication

        :param other: scalar to multiply
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """__mul__: This is not multiplying an TFD by a scalar: it's itemwise:
                    Hack for time evolution.
                    Multiplication by other**L

        :param other: scalar to multiply
        """
        return self.__mul__(1/other)

    def __str__(self):
        return 'Thermofield Double: L={}, d={}, D={}'.format(self.L, sqrt(self.d), sqrt(self.D))

    def random(self, L, d, D, pure=True):
        if pure:
            data = super().random(L, d, D)
            self.data = [kron(a, b) for a, b in zip(data, data)]
            self.d = d**2
            self.D = D**2
        else:
            data = super().random(L, d**2, D)
        return self

    def from_fMPS(self, mps):
        self.data = [kron(a, a) for a in mps.data]
        self.d = mps.d**2
        self.D = mps.D**2
        self.L = mps.L
        return self

    def symmetry(self):
        return fTFD([tra(X.reshape((int(sqrt(X.shape[0])), int(sqrt(X.shape[0])),
                                    int(sqrt(X.shape[1])), int(sqrt(X.shape[1])), 
                                    int(sqrt(X.shape[2])), int(sqrt(X.shape[2])))), 
                         [1, 0, 3, 2, 5, 4]).reshape(X.shape) for X in self])

    def symm_asymm(self, D):
        D = int(sqrt(D))
        return ((eye(D**2) + bd(eye(int(D*(D+1)/2)), -eye(int(D*(D-1)/2))))/2,
                (eye(D**2) - bd(eye(int(D*(D+1)/2)), -eye(int(D*(D-1)/2))))/2,
                bd(eye(int(D*(D+1)/2)), -eye(int(D*(D-1)/2))))

    def get_vL(self):
        prs_vLs = [self.left_null_projector(n, get_vL=True) for n in range(self.L)]
        def vLs(n):
            Pp, Pm, M = self.symm_asymm(self.data[n].shape[1])
            return ((1/2)*prs_vLs[n][1]+(1/2)*M@fs(prs_vLs[n][1]),
                    (1/2)*prs_vLs[n][1]-(1/2)*M@fs(prs_vLs[n][1]), M)
        return [vLs(n) for n in range(self.L)]

    def left_null_projector(self, n, l=None, get_vL=False, store_envs=False, vL=None):
        """left_null_projector:           |
                         - inv(sqrt(l)) - vL = vL- inv(sqrt(l))-
                                               |
        replaces A(n) in TDVP

        :param n: site
        """
        if l is None:
            l, _ = self.get_envs(store_envs)
        if vL is None:
            L_ = sw(cT(self[n])@ch(l(n-1)), 0, 1)
            L = L_.reshape(-1, self.d*L_.shape[-1])
            vL = null(L).reshape((self.d, L.shape[1]//self.d, -1))
        pr = ncon([inv(ch(l(n-1)))@vL, inv(ch(l(n-1)))@c(vL)], [[-1, -2, 1], [-3, -4, 1]])
        if get_vL:
            return pr, vL
        return pr

    def dA_dt(self, H, fullH=False):
        H_ = [(kron(eye(len(h)), h)+kron(h, eye(len(h)))) for h in H]
        dA_dt = super().dA_dt(H_, fullH)
        return dA_dt

    def E(self, op, site):
        """E: one site expectation value

        :param op: 1 site operator
        :param site: site
        """
        op = kron(op, eye(len(op)))
        M = self.mixed_canonicalise(site)[site]
        return trace(sum(cT(M)@tensordot(op, M, [0, 0]), axis=0))

    def Es(self, ops, site):
        ops = [kron(op, eye(len(op))) for op in ops]
        M = self.mixed_canonicalise(site)[site]
        return [trace(sum(cT(M)@tensordot(op, M, [0, 0]), axis=0))
                for op in ops]

class testfTFD(unittest.TestCase):
    def setUp(self):
        """setUp"""
        self.N = N = 4  # Number of MPSs to test
        #  min and max params for randint
        L_min, L_max = 7, 8
        d_min, d_max = 2, 3
        D, D_sq = 3, 9
        # N random MPSs
        self.pure_cases = [fTFD().random(randint(L_min, L_max),
                                         randint(d_min, d_max),
                                         D=D,
                                         pure=True)
                           for _ in range(N)]
        self.mixed_cases = [fTFD().random(randint(L_min, L_max),
                                          randint(d_min, d_max),
                                          D=D_sq,
                                          pure=False)
                           for _ in range(N)]

        psi_0_2 = load('../../tests/fixtures/mat2x2.npy')
        self.tfd_0_2 = fTFD().from_fMPS(fMPS().left_from_state(psi_0_2))

    def test_symmetries(self):
        """test_symmetry"""
        for A, A_ in zip(self.pure_cases, self.mixed_cases):
            self.assertTrue(A==A.symmetry())
            self.assertFalse(A_==A_.symmetry())
            for n in range(A.L):
                vL_sym, vL_asy, M = A.get_vL()[n]
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

                vL_sym, vL_asy, M = A_.get_vL()[n]
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

    def test_time_evolution(self):
        """test_time_evolution"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        H = [Sz1@Sz2+Sx1+Sx2]
        T = linspace(0, 0.1, 2)
        A = self.tfd_0_2
        evs = []
        for _ in range(400):
            evs.append(A.Es([Sx, Sy, Sz], 1))
            A = (A+A.dA_dt(H)*0.1).left_canonicalise()
        plt.plot(evs)
        plt.savefig('x.pdf')
        plt.show()
        
if __name__=='__main__':
    unittest.main(verbosity=2)
