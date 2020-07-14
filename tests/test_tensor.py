import unittest
from numpy.random import randn
from scipy.sparse.linalg import eigs as arnoldi
from scipy.linalg import qr, expm, norm, svd
from numpy.linalg import eig as neig, eigvals, inv
from numpy import swapaxes, count_nonzero, diag, insert, pad, dot, argmax, sqrt
from numpy import allclose, array, tensordot, transpose, all, identity, squeeze
from numpy import isclose, mean, sign, kron, zeros, conj, max, concatenate, eye
from numpy import block, real, imag
import numpy as np
from itertools import product
import scipy as sp
from scipy.linalg import null_space, rq
from xmps.fMPS import fMPS
from xmps.tensor import *
from xmps.spin import spins

from time import time

Sx, Sy, Sz = spins(0.5)
X, Y, Z = 2*Sx, 2*Sy, 2*Sz

class TestTensorTools(unittest.TestCase):
    def setUp(self):
        N = 1
        d = 2
        self.D = D = 5 
        self.tensors = [randn(d, D, D)+1j * randn(d, D, D) for _ in range(N)]

        self.matrices = [randn(D, d*D) + 1j * randn(D, d*D) for _ in range(N)]

        self.matrices_ = [10*(randn(D, D) + 1j * randn(D, D)) for _ in range(N)]
        self.matrices__ = [10*(randn(D, D) + 1j * randn(D, D)) for _ in range(N)]
        self.hermitian_matrices = [10*(A + A.conj().T) for A in self.matrices_]
        self.phases = randn(N) + 1j*randn(N)
        self.phases /= abs(self.phases)
        self.almost_hermitian_matrices = [w*h
                                          for w, h in zip(self.phases,
                                                          self.hermitian_matrices)]

        self.transfer_matrices = [transpose(tensordot(A, C(A), [0, 0]), [0, 2, 1, 3])
                                  for A in self.tensors]
        self.cases = [fMPS().random(10, 2, 5).left_canonicalise() for _ in range(10)]

    def test_mps_pad(self):
        N =10
        rand_Ls = np.random.randint(5, 10, (N,))
        cases = [(fMPS().random(L, 2, np.random.randint(2, 10)).left_canonicalise(),
                  fMPS().random(L, 2, np.random.randint(2, 10)).left_canonicalise())
                 for L in rand_Ls]
        for case1, case2 in cases:
            pre_overlap = case1.full_overlap(case2)
            pre_evs = case1.Es([X, Y, Z], 2)
            print(case1.structure(), case2.structure())
            raise Exception
            case1, case2 = mps_pad(case1, case2)
            self.assertTrue(case1.structure() == case2.structure())
            self.assertAlmostEqual(np.abs(case1.full_overlap(case2)), np.abs(pre_overlap))
            self.assertTrue(np.allclose(pre_evs, case1.Es([X, Y, Z], 2)))

    def test_loc_d(self):
        for A, A_ in zip(self.matrices_, self.matrices__):
            AA = kron(A, A_)
            D = A.shape[0]
            for st, uv in basis_iterator(D):
                self.assertTrue(isclose(AA[loc(st, uv, D)], A[st]*A_[uv]))

    def test_loc_pq(self):
        for A, A_ in zip(self.matrices_, self.matrices):
            AA = kron(A, A_)
            D = A.shape[0]
            d = 2
            for i in range(D):
                for j in range(D):
                    for k in range(D):
                        for l in range(d*D):
                            self.assertTrue(isclose(AA[loc((i, j),(k, l), None, D, d*D)], A[i, j]*A_[k, l]))

    def test_loc_3(self):
        d = 2
        M = randn(d, d)
        H = kron(kron(M, M), M)
        H_ = H.reshape(2, 2, 2, 2, 2, 2)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        for m in range(d):
                            for n in range(d):
                                self.assertTrue(isclose(H[loc_3((i, j), (k, l), (m, n), 2)], M[i, j]*M[k, l]*M[m, n]))

    def test_TT(self):
        for tensor in self.tensors:
            self.assertTrue(allclose(T(T(tensor)), tensor))

    def test_HH(self):
        for tensor in self.tensors:
            self.assertTrue(allclose(H(H(tensor)), tensor))

    def test_get_null_space(self):
        for A in self.matrices:
            vL = get_null_space(A)
            self.assertTrue(allclose(A.dot(vL), 0))
            self.assertTrue(allclose(vL.conj().T.dot(vL), identity(min(vL.shape))))

    def test_r_eigenmatrix(self):
        for M in self.transfer_matrices:
            e, v = r_eigenmatrix(M)
            self.assertTrue(allclose(tensordot(M, v, [[2, 3], [0, 1]]),
                                     e*v))

    def test_l_eigenmatrix(self):
        for M in self.transfer_matrices:
            e, v = l_eigenmatrix(M)
            self.assertTrue(allclose(tensordot(v, M, [[0, 1], [0, 1]]),
                                     e*v))

    def test_l_r_eigenmatrix_same_eigenval(self):
        for M in self.transfer_matrices:
            e_r, _ = r_eigenmatrix(M)
            e_l, _ = l_eigenmatrix(M)
            self.assertTrue(isclose(e_r, e_l))

    def test_transfer_matrices_eigenmatrices_hermitian(self):
        for M in self.transfer_matrices:
            _, v_r = r_eigenmatrix(M)
            _, v_l = l_eigenmatrix(M)
            self.assertTrue(allclose(v_r, H(v_r)))
            self.assertTrue(allclose(v_l, H(v_l)))

    def test_transfer_matrices_eigenmatrices_positive_definite(self):
        for M in self.transfer_matrices:
            _, v_r = r_eigenmatrix(M)
            _, v_l = l_eigenmatrix(M)
            self.assertTrue(all(eigvals(v_r) > 0))
            self.assertTrue(all(eigvals(v_l) > 0))

    def test_rotate_to_hermitian(self):
        for h, h_ in zip(self.hermitian_matrices, self.almost_hermitian_matrices):
            self.assertTrue(allclose(h.conj().T, h))
            self.assertFalse(allclose(h_.conj().T, h_))
            h__ = rotate_to_hermitian(h_)
            self.assertTrue(allclose(h__.conj().T, h__))
            self.assertTrue(allclose(h__, h) or allclose(h__, -h))
            self.assertTrue(allclose(rotate_to_hermitian(h), h))

    def test_split_up(self):
        sz = array([[1, 0], [0, -1]])
        H = kron(sz, sz)
        h = split_up(H, 2)
        for uv, st in basis_iterator(2):
            b, d = uv
            a, c = st
            self.assertTrue(isclose(h[(a, b, c, d)], H[loc(uv, st, 2)]))

        for A, A_ in zip(self.matrices_, self.matrices__):
            AA_ = kron(A, A_)
            AA__ = split_up(AA_, self.D)
            for uv, st in basis_iterator(self.D):
                b, d = uv
                a, c = st
                self.assertTrue(isclose(AA__[(a, b, c, d)], AA_[loc(uv, st, self.D)]))

    def test_split_up_3(self):
        d = 2
        M = randn(d, d)
        H = split_up_3(kron(kron(M, M), M), d)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        for m in range(d):
                            for n in range(d):
                                self.assertTrue(isclose(H[i, j, k, l, m, n], M[i, j]*M[k, l]*M[m, n]))

    def test_unitary_extension(self):
        Qs = [qr(randn(4, 4)+1j*randn(4, 4))[0][:2, :] for _ in range(100)]+\
             [qr(randn(4, 4)+1j*randn(4, 4))[0][:, :2] for _ in range(100)]
        ues = [unitary_extension(Q, 5) for Q in Qs]
        self.assertTrue(allclose([norm(eye(5)-u@u.conj().T) for u in ues], 0))
        self.assertTrue(allclose([norm(eye(5)-u.conj().T@u) for u in ues], 0))

        self.assertTrue(allclose([norm(Q-u[:2, :4]) for Q, u in list(zip(Qs, ues))[:100]], 0))
        self.assertTrue(allclose([norm(Q-u[:4, :2]) for Q, u in list(zip(Qs, ues))[100:]], 0))

    def test_embed_deembed(self):
        v = randn(2, 2)+1j*randn(2, 2)
        self.assertTrue(isclose(norm(deembed(embed(v))-v/norm(v)), 0))

if __name__ == '__main__':
    unittest.main(verbosity=1)
