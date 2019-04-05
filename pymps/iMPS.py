import unittest

from numpy.random import rand, randint, randn
from numpy import diag, dot, tensordot, transpose, allclose
from numpy import real as re, imag as im
from numpy import all, eye, isclose, reshape, swapaxes, trace as tr
from numpy import concatenate, array, stack, sum, identity, zeros, abs 
from numpy import sqrt, real_if_close, around, prod, sign, newaxis
from numpy import concatenate as ct, split as chop, save, load
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

    def __call__(self, k):
        """__call__: 1-based indexing

        :param k: item to get
        """
        return self.data[k+1]

    def __getitem__(self, k):
        """__getitem__: 0-based indexing

        :param k: item to get
        """
        return self.data[k]

    def __setitem__(self, key, value):
        """__setitem__
        """
        self.data[key] = value

    def __eq__(self, other):
        """__eq__

        :param other:
        """
        return array(
            [allclose(a, b) for a, b in zip(self.data, other.data)]).all()

    def __ne__(self, other):
        """__ne__

        :param other:
        """
        return not self.__eq__(other)

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __add__(self, other):
        """__add__: This is not how to add two MPS: it's itemwise addition.
                    A hack for time evolution.

        :param other: MPS with arrays to add
        """
        return iMPS([a+b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        """__sub: This is not how to subtract two MPS: it's itemwise addition.
                    A hack for time evolution.

        :param other: MPS with arrays to subtract
        """
        return iMPS([a-b for a, b in zip(self.data, other.data)])

    def __mul__(self, other):
        """__mul__: This is not multiplying an MPS by a scalar: it's itemwise:
                    Hack for time evolution.
                    Multiplication by other**L

        :param other: scalar to multiply
        """
        return iMPS([other*a for a in self.data])

    def __rmul__(self, other):
        """__rmul__: right scalar multiplication

        :param other: scalar to multiply
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """__mul__: This is not multiplying an MPS by a scalar: it's itemwise:
                    Hack for time evolution.
                    Multiplication by other**L

        :param other: scalar to multiply
        """
        return self.__mul__(1/other)

    def __str__(self):
        return 'iMPS: d={}, D={}'.format(self.d, self.D)

    def copy(self):
        return iMPS(self.data.copy())

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

    def left_canonicalise(self):
        return self.canonicalise('l')

    def right_canonicalise(self):
        return self.canonicalise('r')

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

    def serialize(self, real=False):
        """serialize: return a vector with mps data in it"""
        vec = ct([a.reshape(-1) for a in self])
        if real:
            return ct([vec.real, vec.imag])
        else:
            return vec
        
    def deserialize(self, vec, d, D, p=1, real=False):
        """deserialize: take a vector with mps data (from serialize),
                        make MPS

        :param vec: vector to deserialize
        :param d: local hilbert space dimension
        :param D: bond dimension
        :param p: unit cell
        """
        if real:
            vec = reduce(lambda x, y: x+1j*y, chop(vec, 2))
        self.p, self.d, self.D = p, d, D
        structure = [x for x in self.create_structure(d, D, p)]
        self.data = []
        for shape in structure:
            self.data.append(vec[:prod(shape)].reshape(shape))
            _, vec = chop(vec, [prod(shape)])
        return self

    def store(self, filename):
        """store in file
        :param filename: filename to store in
        """
        save(filename, ct([array([self.d, self.D, self.p]), self.serialize()]))

    def load(self, filename):
        """load from file

        :param filename: filename to load from
        """
        params, arr = chop(load(filename), [3])
        self.d, self.D, self.p = map(lambda x: int(re(x)), params)
        return self.deserialize(arr, self.d, self.D, self.p)

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
