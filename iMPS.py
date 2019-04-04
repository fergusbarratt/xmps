import unittest

from numpy.random import rand, randint, randn
from numpy import diag, dot, tensordot, transpose, allclose, real, imag
from numpy import all, eye, isclose, reshape, swapaxes, trace as tr
from numpy import concatenate, array, stack, sum, identity, zeros, abs
from numpy import sqrt, real_if_close, around, prod, sign, newaxis
from numpy.linalg import cholesky, eigvals, svd, inv, norm
from scipy.sparse.linalg import LinearOperator, eigs as arnoldi
from scipy.linalg import svd as svd_s, cholesky as cholesky_s

from copy import copy

from itertools import product
import matplotlib as mp
import matplotlib.pyplot as plt

from .tensor import H, C, r_eigenmatrix, l_eigenmatrix, get_null_space, p
from .tensor import basis_iterator, T, rotate_to_hermitian
from .spin import spins
Sx, Sy, Sz = spins(0.5)

class TransferMatrix(object):
    """TransferMatrix: Transfer matrix class - implements efficient matrix vector products."""

    def __init__(self, A):
        self.A = A
        self.shape = A.shape[1]**2, A.shape[2]**2
        self.dtype = A.dtype

    def mv(self, r):
        """mv: TM @ v

        :param r: vector to multiply
        """
        A = self.A
        r = r.reshape(A.shape[1:])
        return sum(A @ T(C(A) @ T(r)), axis=0).reshape(prod(A.shape[1:]))

    def mvr(self, l):
        """mvr: TM.H @ v

        :param l: vector to multiply
        """
        A = self.A
        l = l.reshape(A.shape[1:])
        return sum(T(A) @ l @ C(A), axis=0).reshape(prod(A.shape[1:]))

    def aslinearoperator(self):
        """return linear operator representation - for arnoldi etc."""
        return LinearOperator(self.shape, matvec=self.mv, rmatvec=self.mvr)

    def eigs(self, l0=None, r0=None):
        A = self.A
        if l0 is not None:
            l0 = l0.reshape(self.shape[1])
        if r0 is not None:
            r0 = r0.reshape(self.shape[1])
        _,   r = arnoldi(self.aslinearoperator(), k=1, v0=r0)
        eta, l = arnoldi(self.aslinearoperator().H, k=1, v0=l0)

        r, l = (rotate_to_hermitian(r.reshape(A.shape[1:]))/sign(r[0]),
                rotate_to_hermitian(l.reshape(A.shape[1:]))/sign(l[0]))

        n = tr(l @ r)

        return real_if_close(eta), l/sqrt(n), r/sqrt(n)

class iMPS(object):
    """infinite MPS"""

    def __init__(self, data=None, canonical=None):
        """__init__

        :param data: data for matrices in unit cell.
        Expect a list length of the unit cell. canonical describes if data is canonical.
        this is not checked
        """
        if data is not None:
            self.period = len(data)
            self.D = data[0].shape[-1]
            self.d = data[0].shape[0]
            self.data = data
            self.canonical = canonical

    def random(self, d, D, period=1):
        """random: generate d*period normal random matrices of dimension DxD

        :param d: local state space dimension
        :param D: bond dimension
        :param period:
        """
        self.period = period
        self.d = d
        self.D = D
        self.data = [5*(randn(d, D, D)+randn(d, D, D)*1j) for _ in range(period)]
        return self

    def transfer_matrix(self):
        """transfer_matrix"""
        assert self.period == 1
        return [TransferMatrix(A) for A in self.data][0]

    def canonicalise(self, hand='r', l0=None, r0=None, to_vidal=False):
        """canonicalise. See vidal paper.
        This collection of transposes and conjugates makes L_ work
        no idea why. this is weird but it works i think:
        just the vidal canonicalisation procedure with
        lambda the identity and G = A?"""
        A = self.data[0]

        eta, v_l, v_r = self.transfer_matrix().eigs(l0, r0)
        self.l, self.r = v_l, v_r

        X = cholesky_s(v_r, lower=True, check_finite=False, overwrite_a=True)
        Y = cholesky_s(v_l, lower=True, check_finite=False, overwrite_a=True)

        U, L, V = svd_s(Y.T.dot(X), full_matrices=False,
                                    overwrite_a=True,
                                    check_finite=False)
        ll = sum(L*L)

        G_ = (V @ inv(X)) @ A @ (inv(Y.T) @ U)
        L_ = diag(L)

        G_ /= sqrt(eta/ll) # these factors make it so G & L solve the eigenvalue problem
        L_ /= sqrt(ll)     # with eta = 1, and give L specific normalisation.

        self.L = L_  # store singular value

        if hand is 'r':
            self.data[0] = G_ @ L_
        elif hand is 'l':
            self.data[0] = L_ @ G_
        elif hand is 'm':
            sqrtL_ = diag(sqrt(L))
            self.data[0] = sqrtL_ @ G_ @ sqrtL_

        if to_vidal:
            return ivMPS([(G_, L_)])
        else:
            self.canonical = hand
            return self

    def create_structure(self, d, D, p=1):
        return [(d, D, D)]*p

    def eigs(self, l0=None, r0=None):
        """eigs: dominant eigenvectors and values of the transfer matrix."""
        return self.transfer_matrix().eigs(l0, r0)

    def E(self, op, c=None):
        """E: calculate expectation of single site operator

        :param op: operator to compute expectation of
        :param c: canonicalisation of current state.
                  Should be in ['l', 'm', 'r', None].
                  If None, decompose to vidal then use that.
        """
        if c == 'm':
            L = self.L
            A = self.data[0]
            return real_if_close(sum(L @ A * tensordot(op, C(A) @ L, [1, 0])))
        if c == 'r':
            L = self.L
            A = self.data[0]
            return real_if_close(sum(A * tensordot(op, L**2 @ C(A), [1, 0])))
        if c == 'l':
            L = self.L
            A = self.data[0]
            return real_if_close(sum(A @ L**2 * tensordot(op, C(A), [1, 0])))

        if c is None:
            G, L = self.canonicalise(to_vidal=True).data[0]
            circle = tr(G.dot(L).dot(L).dot(H(G)).dot(L).dot(L), axis1=1, axis2=3)
            #  - L - G - L -
            # |      |0     |       |0
            # |    circle   |,      op
            # |      |1     |       |1
            #  - L - G - L -
            return real_if_close(tr(circle @ op))

    def norm(self):
        """norm: should always return 1 since E c=None canonicalises"""
        return self.E(identity(self.d), c=None)



class ivMPS(object):
    """infinite vidal MPS"""

    def __init__(self, data=None):
        """__init__

        :param data: data for matrices in unit cell.
        Expect a list of tuples length of the unit cell
        """
        if data is not None:
            self.period = len(data)
            self.data = data

    def random(self, d, D, period=1):
        """random: generate period normal random tuples
        of matrices of dimension dxDxD and DxD diagonal, resp.

        :param d: local state space dimension
        :param D: bond dimension
        :param period:
        """
        self.period = period
        self.d = d
        self.D = D
        self.data = [((randn(d, D, D)+1j*randn(d, D, D)),
                      diag(sorted(randn(D)**2)[::-1])) for _ in range(period)]
        return self

    def transfer_matrix(self):
        """transfer_matrix"""
        assert self.period == 1
        return [tensordot(A, C(A), (0, 0))
                for A in map(lambda x: dot(*x), self.data)][0]

    def canonicalise(self):
        """canonicalise. See vidal paper.
        This collection of transposes and conjugates makes L_ work
        no idea why"""
        assert self.period == 1
        G, L = self.data[0]
        LG = swapaxes(dot(L, G), 0, 1)
        GL = dot(G, L)

        L_ = transpose(tensordot(LG, C(LG), [0, 0]), [0, 2, 1, 3])
        R_ = transpose(tensordot(GL, C(GL), [0, 0]), [0, 2, 1, 3])

        _, v_l = l_eigenmatrix(L_)
        eta, v_r = r_eigenmatrix(R_)
        v_r = v_r
        v_l = v_l.conj().T

        X = cholesky(v_r)
        Y = cholesky(v_l)

        U, L, V = svd(Y.T.dot(L).dot(X), full_matrices=False)

        L_ = diag(L)

        G_ = transpose(V.dot(inv(X)).dot(G).dot(inv(Y.T)).dot(U),
                       [1, 0, 2])

        LG = swapaxes(dot(L_, G_), 0, 1)
        GL = dot(G_, L_)
        R_ = transpose(tensordot(GL, C(GL), [0, 0]), [0, 2, 1, 3])
        eta, v_r = r_eigenmatrix(R_)

        G_ /= sqrt(eta/sum(L*L))  # these factors make it so G & L solve the eigenvalue problem
        L_ /= sqrt(sum(L*L))      # with eta = 1.

        self.data = [(G_, L_)]
        return self

    def to_iMPS(self):
        """to_iMPS: turn to ordinary MPS (As)"""
        return iMPS(list(map(lambda x: dot(*x), self.data)))

    def E(self, op):
        """E: TOTEST
        """
        G, L = self.data[0]
        circle = tr(G.dot(L).dot(L).dot(H(G)).dot(L).dot(L), axis1=1, axis2=3)
        #  - L - G - L -
        # |      |0     |       |0
        # |    circle   |,      op
        # |      |1     |       |1
        #  - L - G - L -
        return real_if_close(tensordot(circle, op, [[0, 1], [0, 1]]))

class TestTransferMatrix(unittest.TestCase):
    """TestTransferMatrix"""

    def setUp(self):
        N = 5
        D_min, D_max = 9, 10
        d_min, d_max = 2, 3
        p_min, p_max = 1, 2  # always 1 for now
        self.rand_cases = [iMPS().random(randint(d_min, d_max),
                                         randint(D_min, D_max),
                                         period=randint(p_min, p_max))
                           for _ in range(N)]
        self.transfer_matrices = [TransferMatrix(case.data[0])
                                  for case in self.rand_cases]

    def test_transfer_matrices(self):
        for tm in self.transfer_matrices:
            r = rand(tm.shape[1])
            full_tm = transpose(tensordot(tm.A, C(tm.A), [0, 0]), [0, 2, 1, 3]).reshape(tm.shape)
            self.assertTrue(allclose(full_tm @ r, tm.mv(r)))

    def test_aslinearoperator(self):
        for tm in self.transfer_matrices:
            r = rand(tm.shape[1])
            full_tm = transpose(tensordot(tm.A, C(tm.A), [0, 0]), [0, 2, 1, 3]).reshape(tm.shape)
            self.assertTrue(allclose(full_tm @ r, tm.aslinearoperator() @ (r)))

    def test_linear_operator_eigs(self):
        for tm in self.transfer_matrices:
            r = rotate_to_hermitian(arnoldi(tm.aslinearoperator(), k=1)[1].reshape(tm.A.shape[1:]))
            l = rotate_to_hermitian(arnoldi(tm.aslinearoperator().H, k=1)[1].reshape(tm.A.shape[1:]))
            r, l = r/sign(r[0, 0]), l/sign(l[0, 0])
            self.assertTrue(allclose(r, r.conj().T))
            self.assertTrue(allclose(l, l.conj().T))
            self.assertTrue(all(eigvals(r) > 0))
            self.assertTrue(all(eigvals(l) > 0))

class TestiMPS(unittest.TestCase):
    """TestiMPS"""

    def setUp(self):
        N = 5
        D_min, D_max = 10, 11
        d_min, d_max = 2, 3
        p_min, p_max = 1, 2  # always 1 for now
        self.rand_cases = [iMPS().random(randint(d_min, d_max),
                                         randint(D_min, D_max),
                                         period=randint(p_min, p_max))
                           for _ in range(N)]
        self.rand_v_cases = [ivMPS().random(randint(d_min, d_max),
                                            randint(D_min, D_max),
                                            period=randint(p_min, p_max))
                             for _ in range(N)]

    def test_canonicalise_conditions(self):
        for case in self.rand_cases:
            v_case = copy(case).canonicalise(to_vidal=True)  # v returns a vidal form MPS, reuse tests from v_canon
            G, L = v_case.data[0]
            LG = L @ G
            GL = G @ L

            L_ = transpose(tensordot(LG, C(LG), [0, 0]), [0, 2, 1, 3])
            R_ = transpose(tensordot(GL, C(GL), [0, 0]), [0, 2, 1, 3])

            # Normalisation, eigenvalue condition
            self.assertTrue(allclose(tensordot(identity(case.D), L_, [[0, 1], [0, 1]]), identity(case.D)))
            self.assertTrue(allclose(tensordot(R_, identity(case.D), [[2, 3], [0, 1]]), identity(case.D)))

            # L scaled right
            self.assertTrue(isclose(sum(L*L), 1))

            # right canonicalisation
            r_case = copy(case).canonicalise('r')
            A = r_case.data[0]
            self.assertTrue(allclose(tensordot(A, H(A), [[0, -1], [0, 1]]), identity(case.D)))

            # left canonicalisation
            l_case = copy(case).canonicalise('l')
            A = l_case.data[0]
            self.assertTrue(allclose(tensordot(H(A), A, [[0, -1], [0, 1]]), identity(case.D)))

    def test_iMPS_eigs(self):
        for case in self.rand_cases:
            eta, l, r = case.transfer_matrix().eigs()
            self.assertTrue(allclose(tr(l @ r), 1))

    def test_canonicalise_EVs(self):
        for case in self.rand_cases:
            ops = [Sx, Sy, Sz]
            case.canonicalise()
            I_ = case.E(identity(2))
            Ss_ = array([case.E(op) for op in ops])

            # Norm is 1
            self.assertTrue(allclose(I_, 1))

            for _ in range(10):
                case.canonicalise()

            I__ = case.E(identity(2))
            Ss__ = array([case.E(op) for op in ops])

            # Norm conserved, operator expectations don't change
            # After applying 10 canonicalisations
            self.assertTrue(allclose(I__, 1))
            self.assertTrue(allclose(Ss__, Ss_))

    def test_EVs(self):
        for case in self.rand_cases:
            hands = ['l', 'm', 'r', None]
            for hand in hands:
                if hand is not None:
                    case.canonicalise(hand)
                self.assertTrue(isclose(case.E(identity(2), hand), 1))


    def test_v_canonicalise_conditions(self):
        for case in self.rand_v_cases:
            case.canonicalise()
            G, L = case.data[0]
            LG = swapaxes(dot(L, G), 0, 1)
            GL = dot(G, L)

            L_ = transpose(tensordot(LG, C(LG), [0, 0]), [0, 2, 1, 3])
            R_ = transpose(tensordot(GL, C(GL), [0, 0]), [0, 2, 1, 3])

            # Normalisation, eigenvalue condition
            self.assertTrue(allclose(tensordot(identity(case.D), L_, [[0, 1], [0, 1]]), identity(case.D)))
            self.assertTrue(allclose(tensordot(R_, identity(case.D), [[2, 3], [0, 1]]), identity(case.D)))
            # L scaled right
            self.assertTrue(isclose(sum(L*L), 1))

    def test_v_canonicalise_EVs(self):
        for case in self.rand_v_cases:
            ops = [Sx, Sy, Sz]
            case.canonicalise()
            I_ = case.E(identity(2))
            Ss_ = array([case.E(op) for op in ops])

            # Norm is 1
            self.assertTrue(allclose(I_, 1))

            for _ in range(10):
                case.canonicalise()

            I__ = case.E(identity(2))
            Ss__ = array([case.E(op) for op in ops])

            # Norm conserved, operator expectations don't change
            # After applying 10 canonicalisations
            self.assertTrue(allclose(I__, 1))
            self.assertTrue(allclose(Ss__, Ss_))

def suite(iMPS=True):
    suite = []
    if iMPS:
        suite.append(unittest.TestLoader().loadTestsFromTestCase(TestiMPS))
    suite.append(unittest.TestLoader().loadTestsFromTestCase(TestTransferMatrix))
    return unittest.TestSuite(suite)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite(True))
