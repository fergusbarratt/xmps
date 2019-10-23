'''various tensor manipulation tools useful for mps stuff'''
import unittest
from numpy.random import randn
from scipy.sparse.linalg import eigs as arnoldi
from scipy.linalg import qr, expm, norm
from numpy.linalg import eig as neig, eigvals, svd, inv
from numpy import swapaxes, count_nonzero, diag, insert, pad, dot, argmax, sqrt
from numpy import allclose, array, tensordot, transpose, all, identity, squeeze
from numpy import isclose, mean, sign, kron, zeros, conj, max, concatenate, eye
from numpy import block, real, imag
import numpy as np
from itertools import product
import scipy as sp
from scipy.sparse.linalg import LinearOperator, expm_multiply
from scipy.linalg import null_space, rq

from time import time

try:
    from expokitpy import zgexpv, zhexpv
    def lanczos_expm(A, v, t):
        iflag = array([1])
        tol = 0.0
        n = A.shape[0]
        m = min(n//2, 80)
        anorm = 1
        wsp = zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=complex)
        iwsp = zeros(m+2,dtype=int)

        output_vec,tol0,iflag0 = zgexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)
        return output_vec
except:
    def lanczos_expm(A, v, t):
        return expm_multiply(A, v, t)

def embed(v):
    '''put matrix in form
              ↑ ↑
              | |
              ___
               v
              ___
              | |
      '''
    v = v.reshape(1, -1)/norm(v)
    vs = null_space(v).conj().T
    return concatenate([v, vs], 0).T

def deembed(u):
    '''matrix out of form
              ↑ ↑
              | |
              ___
               v   
              ___
              | |
      '''
    return (u@array([1, 0, 0, 0])).reshape(2, 2)

def uqr(A):
    '''Unique qr decomposition'''
    q, r = qr(A, mode='economic')
    O = diag(sign(diag(r)))
    return q@O, O@r

def urq(A):
    '''Unique rq decomposition'''
    r, q = rq(A, mode='economic')
    q_, r_ = qr(A, mode='economic')
    O = diag(sign(diag(r)))
    return r@O, O@q

def eye_like(A):
    """eye_like: identity same shape as A
    """
    return eye(A.shape[0])

def direct_sum(A, B):
    (a1, a2), (b1, b2) = A.shape, B.shape
    O = zeros((a2, b1))
    return block([[A, O], [O.T, B]])
    
def haar_unitary(n):
    return qr(randn(n, n)+1j*randn(n, n))[0]

def unitary_extension(Q, D=None):
    '''extend an isometry to a unitary (doesn't check its an isometry)'''
    s = Q.shape
    flipped=False
    N1 = null_space(Q)
    N2 = null_space(Q.conj().T)
    
    if s[0]>s[1]:
        Q_ = concatenate([Q, N2], 1)
    elif s[0]<s[1]:
        Q_ = concatenate([Q.conj().T, N1], 1).conj().T
    else:
        Q_ = Q

    if D is not None:
        if D > Q_.shape[0]:
            Q_ = direct_sum(Q_, eye(D-Q_.shape[0]))

    return Q_

def ρA(u, keep, dims, optimize=False):
    """Calculate the partial trace of an outer product
    https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    ρ_a = Tr_b(|u><u|)

    Parameters
    ----------
    u : array
        Vector to use for outer product
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    if not keep:
        return np.outer(u, u.conj())
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    u = u.reshape(dims)
    rho_a = np.einsum(u, idx1, u.conj(), idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace
    https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

def T(tensor):
    """T: transpose last two indices of tensor

    :param tensor: tensor to transpose
    """
    return swapaxes(tensor, -1, -2)

def H(tensor):
    """H: Hermitian conjugate of last two indices of a tensor

    :param tensor: tensor to conjugate
    """
    return swapaxes(tensor.conj(), -1, -2)

def C(tensor):
    """C: complex conjugate of tensor

    :param tensor: tensor to conjugate
    """
    return tensor.conj()

def truncate_A(A, S, V, D, minD=True):
    """truncate: truncate A, S, V to D. Ditch all zero diagonals

    :param A: SVD U matrix reshaped
    :param S: SVD S matrix
    :param V: SVD V matrix
    :param D: Bond dimension to truncate to
    """
    if D is None:
        if minD:
            D = rank(S)
        else:
            return (A, S, V)

    A = A[:, :, :D]
    S = S[:D, :D]
    V = V[:D, :]
    return (A, S, V)

def truncate_B(U, S, B, D, minD=True):
    """truncate: truncate U, S, B to D. Ditch all zero diagonals

    :param U: SVD U matrix
    :param S: SVD S matrix
    :param B: SVD V matrix reshaped
    :param D: Bond dimension to truncate to
    """
    if D is None:
        if minD:
            D = rank(S)
        else:
            return (U, S, B)

    return (U[:, :D], S[:D, :D], B[:, :D, :])

def tr_svd(A, D):
    U, s, V = svd(A, full_matrices=False)
    return U[:, :D], diag(s[:D]), V[:D, :]

def diagonalise(M):
    """diagonalise: wrap eig better
       :param M: matrix to diagonalise """
    S, V = neig(M)
    return (V, diag(S))

def rotate_to_hermitian(r, testing=False):
    """rotate_to_hermitian: arnoldi returns matrices that are hermitian up
    to phase w i.e. r = w r_H where r_H = r_H^\dagger and abs(w)=1
    Also slightly underspecified, only return r_H up to a sign
    """
    rr = r[0, 0]
    w = sqrt(conj(rr)/rr)
    return r*w

def r_eigenmatrix(M):
    """eigenmatrix: solve
    -:--|     -|
     M  V = e  V
    -:--|     -|
    for largest (magnitude) e and V.
    :param M: tensor as above: shape (K, K, K, K) for some K.
    right facing indices are 2, 3, left facing 0, 1
    """
    assert all(x == M.shape[0] for x in M.shape)
    assert len(M.shape) == 4
    K = M.shape[0]
    M = M.reshape(K**2, K**2)

    # e_, v_ = neig(M)  # right eig of reshaped M
    # e_max_, v_max_ = e[argmax(abs(e))], v[:, argmax(abs(e))].reshape(K, K)

    e, v = arnoldi(M, k=1, which='LM')
    e_max, v_max = squeeze(e), rotate_to_hermitian(v.reshape(K, K))/sign(v[0])

    return e_max, v_max

def l_eigenmatrix(M):
    """eigenmatrix: solve
    |--:-     |-
    V  M  = e V
    |--:-     |-
    for largest (magnitude) e and V.
    :param M: tensor as above: shape (K, K, K, K) for some K.
    right facing indices are 2, 3, left facing 0, 1
    """
    assert all(x == M.shape[0] for x in M.shape)
    assert len(M.shape) == 4
    K = M.shape[0]
    M = M.reshape(K**2, K**2)

    # e_, v_ = neig(M.conj().T)  # right eig of reshaped, transposed M
    # e_max_, v_max_ = e[argmax(abs(e))], v[:, argmax(abs(e))].reshape(K, K)  # maxima

    e, v = arnoldi(M.conj().T, k=1, which='LM')
    e_max, v_max = squeeze(e), rotate_to_hermitian(v.reshape(K, K))/sign(v[0])

    return e_max, v_max.conj()

def rank(M_diag):
    """rank: return number of non zero elements on diagonal of diagonal matrix

    :param M_diag: diagonal matrix
    """
    return count_nonzero(diag(M_diag))

def mps_pad(mps1, mps2):
    """pad: pad the dimensions of both mps1 and mps2 with zeros such that
    they have the same size

    :param mps1: first mps
    :param mps2: second mps
    """
    padded_mps1, padded_mps2 = mps1, mps2
    if not mps1.structure() == mps2.structure():
        for index, (shapes, (mps1_data, mps2_data)) in enumerate(
                                               zip(zip(mps1.structure(),
                                                       mps2.structure()),
                                                   zip(mps1.data,
                                                       mps2.data))):
            pads = insert(array(shapes[1]) - array(shapes[0]), 0, 0)
            if (pads < 0).any():
                assert not (pads > 0).any()
                pads = list(zip([0, 0, 0], -pads))
                padded_mps2[index] = pad(mps2_data, pads,
                                         mode='constant')
            else:
                pads = list(zip([0, 0, 0], pads))
                padded_mps1[index] = pad(mps1_data, pads,
                                         mode='constant')

    assert structure(padded_mps1) == structure(padded_mps2)
    return (padded_mps1, padded_mps2)

def structure(self):
    """MPS structure"""
    return [x[0].shape for x in self.data]

def ldot(mat, mats):
    """ldot: dot together two tensors on bond indices,
    keeping physical indices first

    :param mat: first set of matrices
    :param mats: second set
    """
    return swapaxes(dot(mat, mats), 0, 1)

def rdot(mats, mat):
    """rdot: dot together two tensors on bond indices,
    keeping physical indices first

    :param mat: first set of matrices
    :param mats: second set
    """
    return dot(mats, mat)

def get_null_space(A):
    """get_null_space: returns matrix of normalised null vectors of A i.e.
    A.vL = 0, v.T.v = I:
    Assumes A.shape = (k, M) with k<M and rank(A)=k

    :param A: matrix to get null space of
    """
    Q, R = qr(H(A))
    return Q[:, A.shape[0]:]

def loc(st, uv, d=None, p=None, q=None):
    """p: return location in tensor product
    for st, uv \in (0, 0), (0, 1), (1, 0), (1, 1) etc.

    :param st: (s, t) -> n
    """
    if d is not None:
        p = q = d
    if d is None:
        if p is None and q is None:
            raise Exception

    st, uv = array(st), array(uv)
    r, s = st
    v, w  = uv
    index = (p*r + v, q*s + w)
    return index

def p(st, uv, d=None):
    return loc(st, uv, d)

def loc_3(rs, tu, vw, d):
    """loc_3: return location in tensor product of 3 elements matrices

    :param st:
    :param uv:
    :param wx:
    """
    if d is not None:
        p = q = d
    if d is None:
        if p is None and q is None:
            raise Exception
    rs, tu, vw = array(rs), array(tu), array(vw)
    r, s = rs
    v, w = vw
    t, u  = tu

    index = (d**2*r + d*v+t, d**2*s + d*w + u)
    return index

def split_up_3(H, d):
    h = zeros((d, d, d, d, d, d)) + 1j*zeros((d, d, d, d, d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    for m in range(d):
                        for n in range(d):
                            h[i, j, k, l, m, n] = H[loc_3((i, j), (k, l), (m, n), 2)]
    return h

def split_up(H, d=None, p=None, q=None):
    """split_up: return (d, d, p, q) element tensor for dp x dq element tensor with elements <u, v|H|s, t> -> (u, v, s, t)
    

    :param H:
    :param d:
    """
    if d is not None:
        p = q = d
    if d is None:
        if p is None and q is None:
            raise Exception

    h = zeros((d, d, p, q)) + 1j*zeros((d, d, p, q))
    for i in range(d):
        for j in range(d):
            for k in range(p):
                for l in range(q):
                    h[i, j, k, l] = H[loc((i, j), (k, l), None, p, q)]
    return transpose(h, [2, 0, 3, 1])

def basis_iterator(d):
    return product(product(range(d), range(d)), product(range(d), range(d)))

def single_basis_iterator(d):
    return product(range(d), range(d))

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
