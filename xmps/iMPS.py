import unittest
from functools import reduce

from numpy.random import rand, randint, randn
from numpy import zeros_like as zl
from numpy import diag, dot, tensordot, transpose, allclose
from numpy import trace as tr, reshape, real, imag, concatenate 
from numpy import real as re, imag as im, copy, swapaxes as sw
from numpy import all, eye, isclose, reshape, swapaxes, trace as tr
from numpy import concatenate, array, stack, sum, identity, zeros, abs 
from numpy import sqrt, real_if_close, around, prod, sign, newaxis
from numpy import concatenate as ct, split as chop, save, load

from numpy.linalg import cholesky, eigvals, svd, inv, norm
from numpy.linalg import inv, cholesky as ch

from scipy.sparse.linalg import LinearOperator, eigs as arnoldi
from scipy.linalg import svd as svd_s, cholesky as cholesky_s
from scipy.linalg import null_space as null
from scipy.optimize import root

from copy import copy

from itertools import product

from .tensor import H, C, r_eigenmatrix, l_eigenmatrix, get_null_space, p
from .tensor import H as cT, C as c, T
from .tensor import basis_iterator, T, rotate_to_hermitian
from .tensor import C as c, H as cT, uqr, urq, eye_like

from .ncon import ncon
from .spin import spins
Sx, Sy, Sz = spins(0.5)
import numpy as np

class Map(object):
    """Map: transfer matrix with A and B"""

    def __init__(self, A, B):
        self.A, self.B = A, B
        self.d, self.D, _ = A.shape
        self.shape = A.shape[1]**2, A.shape[2]**2
        self.dtype = A.dtype

    def full_matrix(self):
        return transpose(tensordot(self.A, 
                                  cT(self.B), [0, 0]), 
                         [0, 2, 1, 3]).reshape(self.shape)

    def mv(self, r):
        """mv: TM @ v

        :param r: vector to multiply
        """
        d, D = self.d, self.D
        A = self.A
        B = self.B
        r = r.reshape(D, D)
        return sum(A@r@cT(B), axis=0).reshape(D**2)

    def mvr(self, l):
        """mvr: v@TM

        :param l: vector to multiply
        """
        d, D = self.d, self.D
        A, B = self.A, self.B
        l = l.reshape(D, D)
        return sum(cT(A)@l@B, axis=0).reshape(D**2)

    def aslinearoperator(self):
        """return linear operator representation - for arnoldi etc."""
        return LinearOperator(self.shape, matvec=self.mv, rmatvec=self.mvr)

    def right_fixed_point(self, r0=None, tol=0):
        d, D = self.d, self.D
        if r0 is not None:
            r0 = r0.reshape(D**2)
        η, r = arnoldi(self.aslinearoperator(), k=1, v0=r0, tol=tol)
        r = rotate_to_hermitian(r)/sign(r[0])
        r = r.reshape(D, D)
        return η*np.sqrt(tr(r.conj().T@r)), r/np.sqrt(tr(r.conj().T@r))

    def left_fixed_point(self, l0=None, tol=0):
        d, D = self.d, self.D
        if l0 is not None:
            l0 = l0.reshape(D**2)
        η, l = arnoldi(self.aslinearoperator().H, k=1, v0=l0, tol=tol)
        l = rotate_to_hermitian(l)/sign(l[0])
        l = l.reshape(D, D)
        return η*np.sqrt(tr(l.conj().T@l)), l/np.sqrt(tr(l.conj().T@l))

    def is_right_eigenvector(self, r, λ=1):
        d, D = self.d, self.D
        r_ = self.aslinearoperator()@r.reshape(D**2)
        return allclose(r, λ*r_.reshape(D, D))

    def is_left_eigenvector(self, l, λ=1):
        d, D = self.d, self.D
        l_ = self.aslinearoperator().H@l.reshape(D**2)
        l_ = l_.reshape(D, D)
        return allclose(l, λ*l_.reshape(D, D))

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
        return sum(A @ r @ cT(A), axis=0).reshape(prod(A.shape[1:]))

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

    def asmatrix(self):
        return transpose(tensordot(self.A, 
                                  c(self.A), [0, 0]), 
                         [0, 2, 1, 3]).reshape(self.shape)

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
        q = 1#tr(l@l)

        return real_if_close(eta), l*sqrt(q)/sqrt(n), r/sqrt(q*n)

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

    @property
    def shape(self):
        return self.data[0].shape

    def copy(self):
        A = iMPS(self.data.copy())
        if hasattr(self, 'r') or hasattr(self, 'l'):
            A.l, A.r = self.l, self.r
        return A

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

    def _lmixed(self, η=1e-10, L0=None):
        '''canonicalisation algorithm described in https://arxiv.org/pdf/1810.07006.pdf'''
        d, D = self.d, self.D
        A = self.data[0]

        if L0 is not None:
            L = L0
        else:
            L = randn(D, D)
        L /= norm(L)

        L_ = copy(L)
        AL, L = uqr((L@A).reshape(d*D, D))
        λ = norm(L)
        L /= λ
        δ = norm(L-L_)
        while δ > η:
            #E = Map(A, AL.reshape(d, D, D))
            #_, L = E.right_fixed_point()
            #_, L = uqr(L)

            L_ = copy(L)

            AL, L = uqr((L@A).reshape(d*D, D))
            λ = norm(L)
            L /= λ
            δ = norm(L-L_)
        AL = AL.reshape(d, D, D)
        return AL, L, λ

    def _rmixed(self, η=1e-10, R0=None):
        '''canonicalisation algorithm described in https://arxiv.org/pdf/1810.07006.pdf'''
        d, D = self.d, self.D
        A = self.data[0]

        if R0 is not None:
            R = R0
        else:
            R = randn(D, D)

        R /= norm(R)

        R_ = copy(R)
        R, AR = urq((A@R).transpose([1, 0, 2]).reshape(D, d*D))
        λ = norm(R)
        R /= λ
        δ = norm(R-R_)
        while δ > η:
            #_, L = Map(A, AL.reshape(d, D, D)).left_fixed_point(l0=L, tol=δ/10)
            #_, L = uqr(L)
            R_ = copy(R)
            R, AR = urq((A@R).transpose([1, 0, 2]).reshape(D, d*D))
            λ = norm(R)
            R /= λ
            δ = norm(R-R_)

        AR = AR.reshape(D, d, D).transpose([1, 0, 2])
        return AR, R, λ

    def mixed(self, η=1e-14):
        '''canonicalisation algorithm described in https://arxiv.org/pdf/1810.07006.pdf'''
        AL, _, λ = self._lmixed(η)
        AR, C, _ = iMPS([AL])._rmixed(η)
        return iMPS([AL]), iMPS([AR]), C

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

    def get_envs():
        _, self.l, self.r = self.eigs()
        return self.l, self.r

    def overlap(self, other):
        A, B = self.left_canonicalise()[0], other.left_canonicalise()[0]
        return np.abs(Map(A, B).left_fixed_point()[0])**2

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

    def Es(self, ops, c=None):
        """E: calculate expectation of single site operator

        :param op: operator to compute expectation of
        :param c: canonicalisation of current state.
                  Should be in ['l', 'm', 'r', None].
                  If None, decompose to vidal then use that.
        """
        ret = []
        if c == 'm':
            L = self.L
            A = self.data[0]
            for op in ops:
                ret.append(real_if_close(sum(L @ A * tensordot(op, C(A) @ L, [1, 0]))))
            return ret
        if c == 'r':
            L = self.L
            A = self.data[0]
            for op in ops:
                ret.append(real_if_close(sum(A * tensordot(op, L**2 @ C(A), [1, 0]))))
            return ret
        if c == 'l':
            L = self.L
            A = self.data[0]
            for op in ops:
                ret.append(real_if_close(sum(A @ L**2 * tensordot(op, C(A), [1, 0]))))
            return ret

        if c is None:
            G, L = self.canonicalise(to_vidal=True).data[0]
            circle = tr(G.dot(L).dot(L).dot(H(G)).dot(L).dot(L), axis1=1, axis2=3)
            #  - L - G - L -
            # |      |0     |       |0
            # |    circle   |,      op
            # |      |1     |       |1
            #  - L - G - L -
            for op in ops:
                ret.append(real_if_close(tr(circle @ op)))
            return ret

    def energy(self, H):
        """energy of sum of two site terms
        """
        d, D = self.d, self.D
        h = H[0].reshape(d, d, d, d)
        A = self.data[0]
        _, l, r = self.eigs()

        C = ncon([h]+[A, A], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA

        K = ncon([l@A.conj(), A.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA
        self.e = tr(K@r)
        return real(self.e)

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
        self.period, self.d, self.D = p, d, D
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
        save(filename, ct([array([self.d, self.D, self.period]), self.serialize()]))

    def load(self, filename):
        """load from file

        :param filename: filename to load from
        """
        params, arr = chop(load(filename), [3])
        self.d, self.D, self.period = map(lambda x: int(re(x)), params)
        return self.deserialize(arr, self.d, self.D, self.period)

    def Lh(self, H, method='krylov', tol=1e-10, testing=False):
        """Lh
        /--|   |--|          |-- 
        l  | h |  |(I-E)^{-1}|  
        \--|   |--|          |--
        """
        _, l, r = self.eigs()
        d, D = self.d, self.D
        h = H[0].reshape(d, d, d, d)
        A = self.data[0]

        def RHS():
            C = ncon([h]+[A, A], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
            K = ncon([l@A.conj(), A.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA
            K -= l*tr(K@r)
            return K

        def O(K, RHS):
            """O: function of which K should be the root
            scipy root doesn't work with complex numbers or ndim matrices (maybe)
            takes a list of length 2*mps.D**2, with first half real parts, second
            half imaginary parts. 
            returns list in the same format

            :param K: guess for K. for correct K will return zeros
            """
            K = reshape(K[:D**2]+1j*K[D**2:], (D, D))

            def LHS(K):
                return K-sum(cT(A)@K@A, axis=0) + l*tr(K@r)

            O_n = reshape(LHS(K) - RHS , (D**2,))
            O_n = concatenate([real(O_n), imag(O_n)])
            return O_n

        K0 = (randn(D, D)+1j*randn(D, D)).reshape(D**2)
        K0 = concatenate([real(K0), imag(K0)])

        K = root(O, K0, args=(RHS(),), method=method, tol=tol).x
        K = reshape(K[:D**2]+1j*K[D**2:], (D, D))
        if testing: 
            def O_(K):
                return O(K, RHS())
            return O_, l, r, K
        return K

    def Rh(self, H, method='hybr', testing=False):
        """Rh
        --|          |--|   |--\ 
          |(I-E)^{-1}|  | h |  r
        --|          |--|   |--/
        """
        _, l, r = self.eigs()
        d, D = self.d, self.D
        h = H[0].reshape(d, d, d, d)
        A = self.data[0]

        def RHS():
            C = ncon([h]+[A, A], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
            K = ncon([A.conj(), A.conj()@r]+[C@r], [[1, -1, 4], [2, 4, 3], [1, 2, -2, 3]]) #AAHAA
            K -= r*tr(l@K)
            return K


        def O(K, RHS):
            """O: function of which K should be the root
            scipy root doesn't work with complex numbers or ndim matrices (maybe)
            takes a list of length 2*mps.D**2, with first half real parts, second
            half imaginary parts. 
            returns list in the same format

            :param K: guess for K. for correct K will return zeros
            """
            K = reshape(K[:D**2]+1j*K[D**2:], (D, D))

            def LHS(K):
                return K-sum(c(A)@K@T(A), axis=0) + r*tr(l@K)

            O_n = reshape(LHS(K) - RHS , (D**2,))
            O_n = concatenate([real(O_n), imag(O_n)])
            return O_n

        K0 = (randn(D, D)+1j*randn(D, D)).reshape(D**2)
        K0 = concatenate([real(K0), imag(K0)])

        K = root(O, K0, args=(RHS(),), method=method, tol=1e-10).x
        K = reshape(K[:D**2]+1j*K[D**2:], (D, D))
        if testing: 
            def O_(K):
                return O(K, RHS())
            return O_, l, r, K
        return K

    def left_null_projector(self, get_vL=False):
        """left_null_projector:           |
                         - inv(sqrt(l)) - vL = vL- inv(sqrt(l))-
                                               |
        replaces A in TDVP
        """
        _, l, _ = self.eigs()
        L_ = sw(cT(self[0])@ch(l), 0, 1)
        L = L_.reshape(-1, self.d*L_.shape[-1])
        vL = null(L).reshape((self.d, L.shape[1]//self.d, -1))

        pr = ncon([inv(ch(l))@vL, inv(ch(l))@c(vL)], [[-1, -2, 1], [-3, -4, 1]])
        self.vL = vL
        if get_vL:
            return pr, vL
        return pr

    def right_null_projector(self, get_vL=False):
        _, _, r = self.eigs()
        R_ = sw(c(self[0])@r, 0, 1)
        R = R_.reshape(-1, self.d*R_.shape[-1])
        vR = sw(null(R).reshape(self.d, R_.shape[-1], -1), 1, 2)
        pr = ncon([inv(ch(r)), vR, c(vR), inv(ch(r))], [[-2, 2], [-1, 1, 2], [-3, 1, 4], [-4, 4]])
        self.vR = vR
        if get_vR:
            return pr, vR
        return pr

    def dA_dt(self, H):
        d, D = self.d, self.D
        h = H[0].reshape(d, d, d, d)
        A = self.data[0]
        _, l, r = self.eigs()

        C = ncon([h]+[A, A], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA

        K = ncon([l@A.conj(), A.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA
        self.e = tr(K@r)

        pr = self.left_null_projector()

        R = ncon([pr, A], [[-3, -4, 1, -2], [1, -1, -5]])
        K = self.Lh(H)

        B = -1j*zl(A)
        B += -1j*ncon([l@C@r, pr, inv(r)@A.conj()], [[3, 4, 1, 2], [-1, -2, 3, 1], [4, -3, 2]])
        B += -1j*ncon([l@A.conj(), pr]+[C], [[1, 3, 4], [-1, -2, 2, 4], [1, 2, 3, -3]])
        B += -1j *ncon([K, R], [[1, 2], [1, 2, -1, -2, -3]])
        B = iMPS([B])
        B.l, B.r, B.vL = l, r, self.vL
        return B

    def update(self, H, δt):
        """mixed gauge update (inverse free) as in verstraeten notes

        :param H: hamiltonian   
        :param δt: timestep
        """
        raise NotImplementedError('Not implemented properly yet')
        self.canonicalise('r')
        d, D = self.d, self.D
        h = H[0].reshape(d, d, d, d)
        A = self.data[0]
        _, l, r = self.eigs()

        C = ncon([h]+[A, A], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
        K = ncon([l@A.conj(), A.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA
        self.e = tr(K@r)

        pr = self.left_null_projector()

        Lh = self.Lh(H)
        Rh = self.Rh(H)

        AL, AR, C = self.mixed()
        AL, AR, C = AL[0], AR[0], C
        AC = AL@C

        G1 = -1j*zl(A)

    
        G1 += ncon([ncon([h]+[AC, AR], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]),
                    c(AR)], [[-1, 2, -2, 1], [2, -3, 1]])
        G1 += ncon([ncon([h]+[AL, AC], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]),
                    c(AL)], [[2, -1, 1, -3], [2, 1, -2]])

        G1 += AC@Rh
        G1 += Lh@AC

        #G2 = ncon([G1, c(AC)], [[2, 1, -1], [2, 1, -2] ])
        #print(tr(G2), self.e)
        #raise Exception

        G2 = ncon([G1, c(AL)], [[2, 1, -1], [2, 1, -2] ])

        print(C1.shape, C2.shape, C3.shape, C4.shape)
        raise Exception

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
