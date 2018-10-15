import unittest
from fMPS import fMPS
from ncon import ncon
from numpy.random import randint, randn
from numpy import kron, swapaxes as sw, eye, transpose as tra, sqrt
from numpy import allclose
from scipy.linalg import norm, null_space as null, inv, cholesky as ch
from scipy.linalg import block_diag as bd
from tensor import H as cT, C as c

def fs(X):
    return X.reshape(int(sqrt(X.shape[0])), int(sqrt(X.shape[0])), *X.shape[1:]
                     ).transpose(1, 0, *list(range(2, len(X.shape)+1))
                     ).reshape(X.shape) 

class fTFD(fMPS):
    def __init__(self, data=None, d=None, D=None):
        """__init__"""
        super().__init__(data, d, D)

    def random(self, L, d, D, pure=True):
        if pure:
            data = super().random(L, d, D)
            self.data = [kron(a, b) for a, b in zip(data, data)]
            self.d = d**2
            self.D = D**2
        else:
            data = super().random(L, d**2, D)
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
        return vLs

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

    def dA_dt(self, H, store_energy=False, fullH=False, prs_vLs=None):
        dA_dt = super().dA_dt(self, H, store_energy=store_energy, fullH=fullH, prs_vLs=prs_vLs)
        for n in range(self.L):
            Pp, Pm, M = symm_asymm(self[n].shape[1])
            dA_dt[n] = (dA_dt[n]+M@fs(dA_dt[n])@M)/2
        return dA_dt


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

    def test_symmetries(self):
        """test_symmetry"""
        for A, A_ in zip(self.pure_cases, self.mixed_cases):
            self.assertTrue(A==A.symmetry())
            self.assertFalse(A_==A_.symmetry())
            for n in range(A.L):
                vL_sym, vL_asy, M = A.get_vL()(n)
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

                vL_sym, vL_asy, M = A_.get_vL()(n)
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

if __name__=='__main__':
    unittest.main(verbosity=2)
