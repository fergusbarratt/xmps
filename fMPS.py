"""fMPS: Finite length matrix product states"""
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import unittest

from numpy.random import rand, randint, randn
from numpy.linalg import svd, inv, norm, cholesky as ch, qr

from numpy import array, concatenate, diag, dot, allclose, isclose, swapaxes as sw
from numpy import identity, swapaxes, trace, tensordot, sum, prod, ones
from numpy import real as re, stack as st, concatenate as ct, zeros, empty
from numpy import split as chop, ones_like, save, load, zeros_like as zl
from numpy import eye, cumsum as cs, sqrt, expand_dims as ed, imag as im
from numpy import transpose as tra, trace as tr, tensordot as td, kron
from numpy import mean

from scipy.linalg import null_space as null, orth, expm#, sqrtm as ch
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from tests import is_right_canonical, is_right_env_canonical, is_full_rank
from tests import is_left_canonical, is_left_env_canonical, has_trace_1

from tensor import H as cT, truncate_A, truncate_B, diagonalise, rank, mps_pad
from tensor import C as c, lanczos_expm, tr_svd, T
from tensor import rdot, ldot, structure
from left_transfer import lt as lt_

from spin import n_body, N_body_spins, spins
from copy import deepcopy, copy
from functools import reduce
from itertools import product
import cProfile
from time import time
import uuid

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz

from ncon import ncon as ncon
#def ncon(*args): return nc(*args, check_indices=False)

class fMPS(object):
    """finite MPS:
    lists of numpy arrays (1d) of numpy arrays (2d). Finite"""

    def __init__(self, data=None, d=None, D=None):
        """__init__

        :param data: data for internal fMPS
        :param d: local state space dimension - can be inferred if d is None
        :param D: Bond dimension: if not none, will truncate with right
        canonicalisation
        """
        self.id = uuid.uuid4().hex # for memoization
        self.id_ = uuid.uuid4().hex # for memoization
        self.id__ = uuid.uuid4().hex # for memoization
        if data is not None:
            self.L = len(data)
            if d is not None:
                self.d = d
            else:
                self.d = data[0].shape[0]
            if D is not None:
                self.right_canonicalise(D)
            else:
                self.D = max([max(x.shape[1:]) for x in data])
            self.data = data

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
        return fMPS([a+b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        """__sub: This is not how to subtract two MPS: it's itemwise addition.
                    A hack for time evolution.

        :param other: MPS with arrays to subtract
        """
        return fMPS([a-b for a, b in zip(self.data, other.data)])

    def __mul__(self, other):
        """__mul__: This is not multiplying an MPS by a scalar: it's itemwise:
                    Hack for time evolution.
                    Multiplication by other**L

        :param other: scalar to multiply
        """
        return fMPS([other*a for a in self.data], self.d)

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

    def left_from_state(self, state):
        """left_from_state: generate left canonical mps from state tensor

        :param state:
        """
        self.L = len(state.shape)
        self.d = state.shape[0]
        Psi = state
        Ls = []

        def split(Psi, n):
            Psi = Psi.reshape(-1, self.d**(self.L-(n+1)), order='F')
            (U, S, V) = svd(Psi, full_matrices=False)
            Ls.append(diag(S))
            return (array(chop(U, self.d)), dot(diag(S), V))

        As = [None]*self.L
        for n in range(self.L):
            (As[n], Psi) = split(Psi, n)
        assert len(As) == self.L

        self.data = As
        self.Ls = Ls

        self.D = max([max(shape[1:]) for shape in self.structure()])

        return self

    def right_from_state(self, state):
        """right_from_state: generate right canonical mps from state tensor

        :param state:
        """
        self.L = len(state.shape)
        self.d = state.shape[0]
        Psi = state
        Ls = []

        def split(Psi, n):
            Psi = Psi.reshape(self.d**(n-1), -1, order='C')
            (U, S, V) = svd(Psi, full_matrices=False)
            Ls.append(diag(S))
            return (dot(U, diag(S)), array(chop(V, self.d, axis=1)))

        Bs = [None]*(self.L+1)
        for n in reversed(range(1, self.L+1)):
            # Exclusive of up_to, generate up_to matrices
            (Psi, Bs[n]) = split(Psi, n)
        Bs = Bs[1:]

        self.data = Bs
        self.Ls = Ls[::-1]

        self.D = max([max(shape[1:]) for shape in self.structure()])

        return self

    def recombine(self):
        state = empty([self.d]*self.L) + 1j*empty([self.d]*self.L)

        for ij in product(*[range(self.d)]*self.L):
            #warnings.simplefilter('ignore')
            state[ij] = reduce(dot, map(lambda k_x: k_x[1][ij[k_x[0]]],
                                        enumerate(self.data)))[0][0]
        return state

    def random(self, L, d, D):
        """__init__

        :param L: Length
        :param d: local state space dimension
        :param D: bond dimension
        generate a random fMPS
        """

        self.L = L
        self.d = d
        fMPS = [rand(*((d,) + shape)) + 1j*rand(*((d,) + shape))
                for shape in self.create_structure(L, d, D)]
        self.D = max([max(shape[1:]) for shape in self.create_structure(L, d, D)])
        self.data = fMPS
        return self

    def create_structure(self, L, d, D):
        """create_structure: generate the structure of a OBC fMPS

        :param L: Length
        :param d: local state space dimension
        :param D: bond dimension
        """

        if D is None:
            D = d**L
        left = [(min(d**n, D), min(d**(n+1), D)) for n in range(L//2)]

        def transpose(ltups):
            l = []
            for tup in ltups:
                l.append((tup[1], tup[0]))
            return l
        if L % 2 == 0:
            structure = left + transpose(left[::-1])
        else:
            structure = left + [(left[-1][-1], transpose(left[::-1])[0][0])] +\
                        transpose(left[::-1])
        return structure

    def right_canonicalise(self, D=None, testing=False, sweep_back=True, minD=True):
        """right_canonicalise: bring internal fMPS to right canonical form,
        potentially with a truncation

        :param D: bond dimension to truncate to during right sweep
        :param testing: test canonicalness
        :param sweep_back: whether to go back and make the left envs the schmidt coefficients
        :param minD: whether to minimize the bond dimension
        """
        if D is not None:
            self.D = min(D, self.D)

        self.ok = True

        def split(M):
            """split: Do SVD, and reshape B matrix

            :param M: matrices
            """
            (U, S, B) = svd(M, full_matrices=False)
            return (U, diag(S), array(chop(B, self.d, axis=1)))

        # left sweep
        for m in range(len(self.data))[::-1]:
            U, S, B = split(concatenate(self[m], axis=1))
            U, S, self[m] = truncate_B(U, S, B, D, minD)
            if m-1 >= 0:
                self[m-1] = self[m-1]@U@S

        if sweep_back and minD:
            # right sweep
            Vs = [None]*self.L
            Ls = [None]*self.L

            V = S = identity(1)
            for m in range(len(self.data)):
                M = trace(dot(dot(dot(dot(cT(self.data[m]),
                                      V),
                                  S),
                              cT(V)),
                          self.data[m]), 0, 0, 2)

                Vs[m], Ls[m] = diagonalise(M)

                self.data[m] = swapaxes(dot(dot(cT(V), self.data[m]), Vs[m]),
                                        0, 1)

                V, S = Vs[m], Ls[m]

            Ls = [ones_like(Ls[-1])] + Ls  # Ones on each end

            for m in range(len(self.data)):
                # sort out the rank (tDj is tilde)
                Dj = Ls[m].shape[0]
                tDj = rank(Ls[m])
                P = concatenate([identity(tDj), zeros((tDj, Dj-tDj))], axis=1)
                self.data[m-1] = self.data[m-1] @ cT(P)
                self.data[m] = P @ self.data[m]
                Ls[m] = P @ Ls[m] @ cT(P)

            self.Ls = Ls  # store all the singular values

        if testing:
            self.update_properties()
            self.ok = self.ok \
                and self.is_right_canonical\
                and self.is_right_env_canonical\
                and self.is_full_rank\
                and self.has_trace_1

        return self

    def left_canonicalise(self, D=None, testing=False, sweep_back=True, minD=True):
        """left_canonicalise: bring internal fMPS to left canonical form,
        potentially with a truncation

        :param D: bond dimension to truncate to during right sweep
        :param testing: test canonicalness
        """
        if D is not None:
            self.D = min(D, self.D)

        self.ok = True

        def split(M):
            """split: Do SVD and reshape A matrix

            :param M: matrix
            """
            (u, S, v) = svd(M)
            A = u[:, :len(S)]
            V = v[:len(S), :]
            o = u[:, len(S):]
            return (array(chop(A, self.d, axis=0)), diag(S), V)

        for m in range(len(self.data)):
            # sort out canonicalisation
            A, S, V = split(concatenate(self.data[m], axis=0))
            self[m], S, V = truncate_A(A, S, V, D, minD)
            if m+1 < len(self.data):
                self[m+1] = S@V@self[m+1]
            else:
                self.norm_ = S@V

        if sweep_back and minD:
            Ls = [None]*self.L
            Vs = [None]*self.L
            V = S = identity(1)

            for m in range(len(self.data))[::-1]:
                # sort out env canonicalisation
                M = trace(dot(dot(dot(dot(self.data[m],
                                      V),
                                  S),
                              cT(V)),
                          cT(self.data[m])), 0, 0, 2)

                Vs[m], Ls[m] = diagonalise(M)

                self.data[m] = swapaxes(dot(dot(cT(Vs[m]), self.data[m]), V),
                                        0, 1)

                V, S = Vs[m], Ls[m]

            Ls.append(ones_like(Ls[0]))  # Ones on each end

            for m in range(len(self.data)):
                # sort out the rank (tDj is tilde)
                Dj = Ls[m].shape[0]
                tDj = rank(Ls[m])
                P = concatenate([identity(tDj), zeros((tDj, Dj-tDj))], axis=1)
                self.data[m-1] = self.data[m-1] @ cT(P)
                self.data[m] = P @ self.data[m]
                Ls[m] = P @ Ls[m] @ cT(P)

            self.Ls = Ls  # store all the singular values

        if testing:
            self.update_properties()
            self.ok = self.ok\
                and self.is_left_canonical\
                and self.is_left_env_canonical\
                and self.is_full_rank\
                and self.has_trace_1

        return self

    def mixed_canonicalise(self, oc, D=None, testing=False):
        """mixed_canonicalise: bring internal fMPS to mixed canonical form with
        orthogonality center oc, potentially with a truncation

        :param oc: orthogonality center
        :param D: bond dimension
        :param testing: test canonicalness
        """
        self.ok = True
        self.oc = oc
        self.right_canonicalise(D, minD=False)

        def split(M):
            """split: Do SVD and reshape A matrix

            :param M: matrix
            """
            (A, S, V) = svd(M, full_matrices=False)
            return (array(chop(A, self.d, axis=0)), diag(S), V)

        for m in range(len(self.data))[:oc]:
            # sort out canonicalisation
            A, S, V = split(concatenate(self.data[m], axis=0))
            self.data[m], S, V = truncate_A(A, S, V, D)
            if m+1 < len(self.data):
                self[m+1] = S@V@self[m+1]

        if testing:
            self.ok = self.ok and is_left_canonical(self.data[:oc])\
                              and is_right_canonical(self.data[oc+1:])

        return self

    def entanglement_entropy(self, site):
        """von neumann entanglement entropy across bond to left of site
        """
        self.left_canonicalise()
        return -self.Ls[site]@log(self.Ls[site])

    def update_properties(self):
        """update_properties: Could be slow"""
        self.is_left_canonical = is_left_canonical(self.data)
        self.is_right_canonical = is_right_canonical(self.data)
        self.has_norm_1 = isclose(self.norm(), 1)
        try:
            self.is_left_env_canonical = is_left_env_canonical(self.data,
                                                               self.Ls)
            self.is_right_env_canonical = is_right_env_canonical(self.data,
                                                                 self.Ls)
            self.is_full_rank = is_full_rank(self.Ls)
            self.has_trace_1 = has_trace_1(self.Ls)
        except AttributeError:
            self.is_left_env_canonical = False
            self.is_right_env_canonical = False
            self.is_full_rank = False
            self.has_trace_1 = False

    def structure(self):
        """structure"""
        return [x[0].shape for x in self.data]

    def overlap(self, other):
        """overlap

        :param other: other with which to calculate overlap
        """
        assert len(self) == len(other)

        padded_self, padded_other = mps_pad(self, other)

        F = ones((1, 1))
        for L, R in zip(padded_self, padded_other):
            F = tensordot(tensordot(cT(L), F, (-1, 1)), R, ([0, -1], [0, 1]))
        return F[0][0]

    def norm(self):
        """norm: not efficient - computes full overlap.
        use self.E(identity(self.d), site) for mixed
        canonicalising version"""
        return self.overlap(self)

    def get_envs(self, store_envs=False):
        """get_envs: return all envs (slow). indexing is a bit weird: l(-1) is |=
           returns: fns tuple of functions l, r"""
        def get_l(mps):
            ls = [array([[1]])]
            for n in range(len(mps)):
                ls.append(sum(re(cT(mps[n]) @ ls[-1] @ mps[n]), axis=0))
            return lambda n: ls[n+1]

        def get_r(mps):
            rs = [array([[1]])]
            for n in range(len(mps))[::-1]:
                rs.append(sum(re(mps[n] @ rs[-1] @ cT(mps[n])), axis=0))
            return lambda n: rs[::-1][n+1]

        if store_envs:
            self.l, self.r = get_l(self), get_r(self)
            return self.l, self.r

        return get_l(self), get_r(self)

    def left_transfer(self, op, j, i, ret_all=True):
        """transfer an operator (u, d, ...) on aux indices at i to site(s) j
        Returns a function such that R(j) == op at j. No bounds checking.
        """
        def lt(op, As, j, i):
            Ls = [op]
            for m in reversed(range(j, i)):
                W = td(As[m], td(As[m].conj(), Ls[0], [2, 1]), [[0, 2], [0, 2]])
                Ls.insert(0, W)
            return Ls

        Ls = lt_(op, self.data, j, i)
        return (lambda n: Ls[n-j]) if ret_all else Ls[0]

    def right_transfer(self, op, i, j, ret_all=True):
        """transfer an operator (..., u, d) on aux indices at i to site(s) j
        Returns a function such that R(j) == op at j. No bounds checking.
        """
        Rs = [op]
        oplinks = list(range(-1, -len(op.shape)+1, -1))+[2, 3]
        for m in range(i+1, j+1):
            Rs.append(ncon([self[m].conj(), self[m], Rs[-1]], [[1, 3, -len(op.shape)+1], [1, 2, -len(op.shape)], oplinks]))
        return (lambda n: Rs[n-i-1]) if ret_all else Rs[0]

    def links(self, op=True):
        """links: op True: return the links for ncon for full contraction of this
        mps with operator shape of [d,d]*L
                  op False: return links for full contraction of this mps
        """
        if op:
            L = self.L
            links = [[n, 2*L+n, 2*L+n+1] for n in range(1, L+1)]+[list(range(1, 2*L+1))]+\
                    [[L+n, 3*L+n if n!=1 else 2*L+n , 3*L+n+1 if n!=L else 2*L+n+1] for n in range(1, L+1)]
            return links
        else:
            L = self.L
            links = [[n, L+n, L+n+1] for n in range(1, L+1)]+[[n, (1+int(n!=1))*L+n, (1+int(n!=L))*L+n+1] for n in range(1, L+1)]
            return links

    def apply(self, opsite):
        op, site = opsite
        self[site] = td(op, self[site], [1, 0])
        return self

    def copy(self):
        return fMPS(self.data.copy())

    def E(self, op, site):
        """E: one site expectation value

        :param op: 1 site operator
        :param site: site
        """
        M = self.mixed_canonicalise(site)[site]
        return re(tensordot(op, trace(dot(cT(M),  M), axis1=1, axis2=3), [[0, 1], [0, 1]]))

    def Es(self, ops, site):
        M = self.mixed_canonicalise(site)[site]
        return [re(tensordot(op, trace(dot(cT(M),  M), axis1=1, axis2=3), [[0, 1], [0, 1]]))
                for op in ops]

    def E_L(self, op):
        """E_L: expectation of a full size operator

        :param op: operator
        """
        op = op.reshape([self.d, self.d]*self.L)
        return re(ncon([a.conj() for a in self]+[op]+self.data, self.links()))

    def energy(self, H, fullH=False):
        """energy: if fullH: alias for E_L, else does nn stuff

        :param H: hamiltonian
        """
        if not fullH:
            l, r = self.get_envs()
            e = []
            for m, h in enumerate(H):
                h = h.reshape(2, 2, 2, 2)
                C = ncon([h]+self.data[m:m+2], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]])
                e.append(ncon([c(l(m-1))@self.data[m].conj(), self.data[m+1].conj()@c(r(m+1))]+[C], [[1, 3, 4], [2, 4, 5], [1, 2, 3, 5]]))
            return re(sum(e))
        else:
            return self.E_L(H)

    def serialize(self, real=False):
        """serialize: return a vector with mps data in it"""
        vec = concatenate([a.reshape(-1) for a in self])
        if real:
            return ct([vec.real, vec.imag])
        else:
            return vec

    def deserialize(self, vec, L, d, D, real=False):
        """deserialize: take a vector with mps data (from serialize),
                        make MPS

        :param vec: vector to deserialize
        :param L: length of mps to make
        :param d: local hilbert space dimension
        :param D: bond dimension
        """
        if real:
            vec = reduce(lambda x, y: x+1j*y, chop(vec, 2))
        self.L, self.d, self.D = L, d, D
        structure = [(d, *x) for x in self.create_structure(L, d, D)]
        self.data = []
        for shape in structure:
            self.data.append(vec[:prod(shape)].reshape(shape))
            _, vec = chop(vec, [prod(shape)])
        return self

    def store(self, filename):
        """store in file
        :param filename: filename to store in
        """
        save(filename, ct([array([self.L, self.d, self.D]), self.serialize()]))

    def load(self, filename):
        """load from file

        :param filename: filename to load from
        """
        params, arr = chop(load(filename), [3])
        self.L, self.d, self.D = map(lambda x: int(re(x)), params)
        return self.deserialize(arr, self.L, self.d, self.D)

    def dA_dt(self, H, store_energy=False, fullH=False, prs=None):
        """dA_dt: Finds A_dot (from TDVP) [B(n) for n in range(n)], energy. Uses inverses.
        Indexing is A[0], A[1]...A[L-1]

        :param self: matrix product state @ current time
        :param H: Hamiltonian
        """
        self.dA_cache = {}
        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs(True)
        prs = [self.left_null_projector(n, l) for n in range(self.L)] if prs is None else prs
        def pr(n): return prs[n]

        if not fullH:
            def B(i, H=H):
                e = []
                B = -1j*zl(A[i])
                _, Dn, Dn_1 = A[i].shape
                H = [h.reshape(2, 2, 2, 2) for h in H]

                if d*Dn==Dn_1:
                    # Projector is full of zeros
                    return -1j*B

                R = ncon([pr(i), A[i]], [[-3, -4, 1, -2], [1, -1, -5]])
                Rs = self.left_transfer(R, 0, i) # list of E**k @ R - see left_transfer docs
                self.dA_cache[str(i)] = Rs

                for m, h in reversed(list(enumerate(H))):
                    if m > i:
                        # from gauge symmetry
                        continue

                    Am, Am_1 = self.data[m:m+2]
                    C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
                    K = ncon([l(m-1)@Am.conj(), Am_1.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA
                    e.append(trace(K@r(m+1)))
                    #C -= trace(K@r(m+1))
                    #K -= trace(K@r(m+1))

                    if m==i:
                        B += -1j *ncon([l(m-1)@C@r(m+1), pr(m), inv(r(m))@Am_1.conj()], [[3, 4, 1, 2], [-1, -2, 3, 1], [4, -3, 2]])
                    if m==i-1:
                        B += -1j *ncon([l(m-1)@Am.conj(), pr(m+1)]+[C], [[1, 3, 4], [-1, -2, 2, 4], [1, 2, 3, -3]])
                    if m < i-1:
                        B += -1j *ncon([K, Rs(m+2)], [[1, 2], [1, 2, -1, -2, -3]])
                self.e = e
                return B
        else:
            def B(n, H=H):
                H = H.reshape([self.d, self.d]*self.L)
                links = self.links()
                # open up links for projector
                links[n] = [-1, -2]+links[n][:2]
                if n == L-1:
                    links[-1] = links[-1][:2]+[-3]
                else:
                    links[n+1] = [links[n+1][0], -3, links[n+1][2]]
                return -1j *ncon([pr(m) if m==n else c(inv(r(n)))@a.conj() if m==n+1 else a.conj() for m, a in enumerate(A)]+[H]+A, links)

        return fMPS([B(i) for i in range(L)])

    def projection_error(self, H, dt):
        """projection_error: error in projection to mps manifold

        :param H: hamiltonian
        :param dt: time step
        """
        L, d, A, D = self.L, self.d, self.data, self.D
        dA_dt = self.dA_dt(H, True)
        l, r = self.l, self.r
        def vR(n): return self.right_null_projector(n, r, True)[1]

        H = [h.reshape(2, 2, 2, 2) for h in H]
        def G(m):
            C = ncon([H[m]]+[l(m-1)@A[m], A[m+1]@r(m+1)], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
            return ncon([c(vL(m)), c(vR(m+1)), C], [[1, 3, -1], [2, -2, 4], [1, 2, 3, 4]])

        ε = 0
        for m, (Dm, Dm_1) in list(enumerate(self.structure()))[:-1]:
            _, Dm_2 = self.structure()[m+1]
            if d*Dm==Dm_1 or Dm_1==d*Dm_2:
                continue
            def vL(n): return self.left_null_projector(n, l, True)[1]
            U, s, V = tr_svd(G(m), d*D-D)

            x, y = U@sqrt(s), sqrt(s)@V
            dA01, dA10 = sqrt(dt)*l(m-1)@vL(m)@x, sqrt(dt)*y@vR(m+1)@r(m+1)

            ε += re(tr(sum(cT(dA01)@l(m-1)@dA01, 0)@sum(dA10@r(m+1)@cT(dA10), 0)))

            ## left environments change as we move along the chain
            # this is a hack because I used lambdas for the envs in vL etc.
            ln = sum(cT(A[m])@l(m-1)@A[m], 0)
            def l(n): return ln
        self.ε = ε # store last projection error on mps
        return ε

    def should_expand(self, H, dt, threshold):
        """should_expand the manifold?
        """
        return self.projection_error(H, dt) > threshold

    def expand(self, D_):
        """expand - just pad with zeros"""
        from numpy import pad
        L, d, D = self.L, self.d, self.D
        for m, (sh, sh_) in enumerate(zip(self.structure(), self.create_structure(L, d, D_))):
            self[m] = pad(self[m], list(zip([0, 0, 0], (0, *tuple(array(sh_)-array(sh))))), 'constant')
        self.D = D_
        return self.left_canonicalise(minD=False)

    def dynamical_expand(self, H, dt, D_, threshold=1e-8):
        """dynamical_expand: expand bond dimension to D_ during timestep dt with H

        :param H: hamiltonian
        :param dt: timestep
        :param D_: new bond dimension
        :param threshold: by default will not expand if no need to:
                          set to None to expand regardless
        """
        L, d, A, D = self.L, self.d, self.data, self.D
        def vR(n): return self.right_null_projector(n, r, True)[1]

        H = [h.reshape(2, 2, 2, 2) for h in H]
        def G(m):
            return ncon([l(m-1)@A[m], A[m+1]@r(m+1), H[m], c(vL(m)), c(vR(m+1))],
                        [[2, 1, 4], [6, 4, 8], [3, 6, 2, 7], [3, 1, -1], [7, -2, 8]])

        if threshold is not None:
            if self.projection_error(H, dt) < threshold:
                return self

        for m, (Dm, Dm_1) in list(enumerate(self.structure()))[:-1]:
            _, Dm_2 = self.structure()[m+1]
            if d*Dm==Dm_1 or Dm_1==d*Dm_2:
                continue
            def vL(n): return self.left_null_projector(n, l, True)[1]
            dA_dt = self.dA_dt(H)
            l, r = self.l, self.r

            U, s, V = tr_svd(G(m), D_-D)
            x, y = U@sqrt(s), sqrt(s)@V
            dA01, dA10 = sqrt(dt)*l(m-1)@vL(m)@x, sqrt(dt)*y@vR(m+1)@r(m+1)

            A[m], A[m+1] = ct([A[m]+dt*dA_dt[m], dA01], 2), ct([A[m+1]+dt*dA_dt[m+1], dA10], 1)

        self.D = max([max(shape) for shape in self.structure()])

        return self

    def grow(self, H, dt, D_):
        """grow bond dimension to D_ with timesteps dt. goes d*D at a time
        """
        if D_ is None:
            return self
        while self.D<min(D_, (self.d**(self.L//2))):
            self.dynamical_expand(H, dt, min(self.d*self.D, D_), None)
        return self

    def ddA_dt(self, v, H, fullH=False):
        """d(dA)_dt: find ddA_dt - in 2nd tangent space:
           for time evolution of v in tangent space

        :param H: hamiltonian
        :param fullH: is the hamiltonian full
        """
        L, d, A = self.L, self.d, self.data
        dA = self.import_tangent_vector(v)
        dA_dt = self.dA_dt(H, store_energy=True, fullH=fullH)
        l, r, e = self.l, self.r, self.e
        # down are the tensors acting on c(Aj), up act on Aj
        up, down = self.jac(H, False)

        ddA = [sum([td(down(i, j), c(dA[j]), [[3, 4, 5], [0, 1, 2]]) +
                    td(up(i, j),     dA[j],  [[3, 4, 5], [0, 1, 2]]) for j in range(L)], axis=0)
               for i in range(L)]

        return self.extract_tangent_vector(ddA)

    def jac(self, H,
            as_matrix=True,
            real_matrix=True,
            as_linearoperator=False,
            fullH=False,
            testing=False,
            parallel_transport=True):
        """jac: calculate the jacobian of the current mps
        """
        L, d, A = self.L, self.d, self.data
        dA_dt = self.dA_dt(H, fullH=fullH)
        l, r = self.l, self.r
        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)]
        prs = [x[0] for x in prs_vLs]
        vLs = [x[1] for x in prs_vLs]
        def vL(i): return vLs[i]

        # Get tensors
        ## unitary rotations: -<d_iψ|(d_t |d_kψ> +iH|d_kψ>) (dA_k)
        #-<d_iψ|d_kd_jψ> dA_j/dt (dA_k) (range(k+1, L))
        #def Γ1(i, k): return sum([td(c(self.christoffel(k, j, i, envs=(l, r))), l(j-1)@dA_dt[j]@r(j), [[3, 4, 5], [0, 1, 2]]) for j in range(L)], axis=0)
        #-i<d_iψ|H|d_kψ> (dA_k)
        id = uuid.uuid4().hex # for memoization
        def F1(i, k): return  -1j*self.F1(i, k, H, envs=(l, r), prs_vLs=prs_vLs, fullH=fullH, id=id)

        def ungauge_i(tens, i, conj=False):
            def co(x): return x if not conj else c(x)
            k = len(tens.shape[3:])
            links = [1, 2, -2]+list(range(-3, -3-k, -1))
            return ncon([ch(l(i-1))@co(vL(i)),
                        ncon([tens, ch(r(i))], [[-1, -2, 1, -4, -5, -6], [1, -3]])],
                        [[1, 2, -1], links])
        def ungauge_j(tens, j, conj=False):
            def co(x): return x if not conj else c(x)
            k = len(tens.shape[:-3])
            links = list(range(-1, -k-1, -1))+[1, 2, -k-2]
            return ncon([ch(l(j-1))@co(vL(j)), tens@ch(r(j))],
                    [[1, 2, -k-1], links])
        def ungauge(tens, i, j, conj=(True, False)):
            c1, c2 = conj
            return ungauge_j(ungauge_i(tens, i, c1), j, c2)

        ## non unitary (from projection): -<d_id_kψ|(d_t |ψ> +iH|ψ>) (dA_k*) (should be zero for no projection)
        #-<d_id_kψ|d_jψ> dA_j/dt (dA_k*)
        def Γ2(i, k): return ungauge(self.christoffel(i, k, min(i, k), envs=(l, r), prs_vLs=prs_vLs, closed=(None, None, l(min(i, k)-1)@dA_dt[min(i, k)]@r(min(i, k)))),
                                     i, k, conj=(True, True))
        #-i<d_id_k ψ|H|ψ> (dA_k*)
        def F2(i, k): return -1j*ungauge(self.F2(i, k, H, envs=(l, r), prs_vLs=prs_vLs, fullH=fullH, id=id),
                                         i, k, conj=(True, True))

        def F2t(i, j): return F2(i, j) + Γ2(i, j) #F2, Γ2 act on dA*j

        vLs, sh = self.tangent_space_dims(l, True)
        if not as_matrix:
            def gauge(G, i, j):
                return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@c(vL(j)), inv(ch(r(i))), inv(ch(r(j)))], 
                            [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])
            def gauge_(G, i, j):
                return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@vL(j), inv(ch(r(i))), inv(ch(r(j)))], 
                            [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])

            return (lambda i, j: gauge(F1(i, j), i, j)), (lambda i, j: gauge_(F2t(i, j), i, j))

        nulls = len([1 for (a, b) in sh if a==0 or b==0])
        shapes = list(cs([prod([a, b]) for (a, b) in sh if a!=0 and a!=0]))
        DD = shapes[-1]
        # these apply l-vL- -r- to something like -A|- to  get something like =x-
        def ind(i):
            slices = [slice(a[0], a[1], 1)
                      for a in [([0]+shapes)[i:i+2] for i in range(len(shapes))]]
            return slices[i]

        J1_ = -1j*zeros((DD, DD))
        J2_ = -1j*zeros((DD, DD))
        for i_ in range(len(shapes)):
            for j_ in range(len(shapes)):
                i, j = i_+nulls, j_+nulls
                J1_ij = F1(i,j)
                J2_ij = F2t(i,j)
                J1_[ind(i_), ind(j_)] = J1_ij.reshape(prod(J1_ij.shape[:2]), -1)
                J2_[ind(i_), ind(j_)] = J2_ij.reshape(prod(J2_ij.shape[:2]), -1)

        if not real_matrix:
            return J1_, J2_

        J = kron(Sz, re(J2_)) + kron(eye(2), re(J1_)) + kron(Sx, im(J2_)) + kron(-1j*Sy, im(J1_))
        if as_linearoperator:
            return aslinearoperator(J)
        else:
            return J

    def F1(self, i_, j_, H, envs=None, prs_vLs=None, fullH=False, testing=False, id=None):
            '''<d_iψ|H|d_jψ>
               Does some pretty dicey caching stuff to avoid recalculating anything'''
            # if called with new id, need to recompute everything
            # otherwise, we should look in the cache for computed values
            id = id if id is not None else uuid.uuid4().hex
            if self.id != id:
                self.id = id
                # initialize the memories 
                # we only don't try the cache on the first call from jac
                self.F1_i_mem_ = {}
                self.F1_j_mem_ = {}
                self.F1_tot_ij_mem = {}
            else:
                # read from cache: 
                # have we cached this tensor?
                if str(i_)+str(j_) in self.F1_tot_ij_mem:
                    return self.F1_tot_ij_mem[str(i_)+str(j_)]

                ## have we cached its conjugate?
                if str(j_)+str(i_) in self.F1_tot_ij_mem:
                    return c(tra(self.F1_tot_ij_mem[str(j_)+str(i_)], 
                                 [2, 3, 0, 1]))

            L, d, A = self.L, self.d, self.data
            l, r = self.get_envs() if envs is None else envs
            prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)] if prs_vLs is None else prs_vLs
            def pr(n): return prs_vLs[n][0]
            def vL(n): return prs_vLs[n][1]

            if not fullH:
                i, j = (j_, i_) if j_<i_ else (i_, j_)
                gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
                gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1
                G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))
                d, Din_1, Di = self[i].shape
                icr, icl = [inv(ch(r(i))) for i in range(self.L)], [inv(ch(l(i))) for i in range(self.L)]
                cr, cl = [ch(r(i)) for i in range(self.L)], [ch(l(i)) for i in range(self.L)]
                def inv_ch_r(n): return icr[n]
                def inv_ch_l(n): return icl[n]
                def ch_r(n): return cr[n]
                def ch_l(n): return cl[n]

                if not d*Din_1==Di:
                    H = [h.reshape(2, 2, 2, 2) for h in H]
                    if str(i) not in self.F1_i_mem_:
                        # compute i properties, and store in cache
                        Rd_ = ncon([inv_ch_l(i-1)@c(vL(i)), A[i]], [[1, -2, -3], [1, -1, -4]])
                        Lbs_ = self.right_transfer(ncon([inv_ch_r(i), inv_ch_r(i)], [[-1, -3], [-2, -4]]), i, L-1)
                        Rds_ = self.left_transfer(Rd_, 0, i)

                        self.F1_i_mem_[str(i)] = (Lbs_, Rds_)
                    else:
                        # read i properties from cache
                        Lbs_, Rds_ = self.F1_i_mem_[str(i)]

                    if str(j) not in self.F1_j_mem_:
                        # compute j properties, and store in cache
                        Ru_ = ncon([inv_ch_l(j-1)@vL(j), c(A[j])@ch_r(j)], [[1, -1, -3], [1, -2, -4]])
                        Rb_ = ncon([inv_ch_l(j-1)@c(vL(j)), inv_ch_l(j-1)@vL(j)], [[1, -1, -3], [1, -2, -4]])
                        Rus_ = self.left_transfer(Ru_, 0, j)
                        Rbs_ = self.left_transfer(Rb_, 0, j)

                        self.F1_j_mem_[str(j)] = (Rus_, Rbs_)
                    else:
                        # read j properties from cache
                        Rus_, Rbs_ = self.F1_j_mem_[str(j)]

                    for m, h in reversed(list(enumerate(H))):
                        if m > i:
                            # from gauge symmetry
                            if not i==j:
                                continue
                            else:
                                Am, Am_1 = self.data[m:m+2]
                                Kr_ = ncon([Am_1@r(m+1), c(Am_1), Am, c(Am), h], [[2,4,1], [3,5,1], [6,-1,4], [7,-2,5], [7,3,6,2]])
                                #AAHAA
                                Lbs__ = sum(cT(vL(i))@vL(i), axis=0)
                                G_ += ncon([tr(Lbs_(m)@Kr_, 0, -1, -2), Lbs__], [[-2, -4], [-1, -3]])
                        Am, Am_1 = self.data[m:m+2]
                        if m==i:
                            if j==i:
                                # BAHBA
                                G_ += ncon([vL(m), inv_ch_r(m)@Am_1@r(m+1)]+[h]+[c(vL(m)), inv_ch_r(m)@c(Am_1)], 
                                           [[5, 1, -3], [6, -4, 2], [5, 6, 3, 4], [3, 1, -1], [4, -2, 2]])

                            elif j==i+1:
                                # ABHBA
                                G_ += ncon([Am, inv_ch_l(m)@vL(m+1)]+[h]+[ch_l(m-1)@c(vL(m)), inv_ch_r(m)@c(Am_1)@ch_r(m+1)], 
                                           [[5, 1, 2], [6, 2, -3], [5, 6, 3, 4], [3, 1, -1], [4, -2, -4]])
                            else:
                                # AAHBA
                                O_ = ncon([ch(l(m-1))@Am, Am_1]+[h]+[c(vL(m)), inv_ch_r(m)@c(Am_1)], 
                                          [[3, 1, 2], [4, 2, -3], [3, 4, 5, 6], [5, 1, -1], [6, -2, -4]])

                                G_+= tensordot(O_, Rus_(m+2), [[-1, -2], [0, 1]])
                        elif m==i-1:
                            if j==i:
                                # ABHAB
                                G_ += ncon([l(m-1)@Am, inv_ch_l(m)@vL(m+1)]+[h]+[c(Am), inv_ch_l(m)@c(vL(m+1))]+[eye(r(m+1).shape[0])],
                                           [[7, 1, 2], [8, 2, -3], [5, 6, 7, 8], [5, 1, 3], [6, 3, -1], [-2, -4]])

                            else:
                                # AAHAB
                                Q_ = ncon([l(m-1)@Am, Am_1]+[h]+[c(Am), inv_ch_l(m)@c(vL(m+1))], 
                                          [[4, 1, 2], [5, 2, -2], [4, 5, 6, 7], [6, 1, 3], [7, 3, -1]]) 

                                G_ += tensordot(Q_, ncon([inv_ch_r(m+1), Rus_(m+2)], [[-2, 1], [-1, 1, -3, -4]]), [-1, 0])
                        elif m<i:
                            C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
                            K = ncon([l(m-1)@c(Am), c(Am_1)]+[C],
                                     [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]])
                            #AAHAA
                            if i==j:
                                G_ += ncon([tensordot(K, Rbs_(m+2), [[0, 1], [0, 1]]), eye(r(j).shape[0])], [[-1, -3], [-2, -4]])
                            else:
                                G_ += tensordot(tensordot(K, Rds_(m+2), [[0, 1], [0, 1]]), 
                                                ncon([Rus_(i+1), inv_ch_r(i)], [[-1, 2, -3, -4], [2, -2]]), [-1, 0])

                if testing:
                    G = 1j*zeros((*A[i].shape, *A[j].shape))
                    d, Din_1, Di = self[i].shape
                    H = [h.reshape(2, 2, 2, 2) for h in H]

                    Rd = ncon([pr(i), A[i]], [[-3, -4, 1, -2], [1, -1, -5]])
                    Lb = ncon([l(i-1)]+[pr(i), pr(i), inv(r(i)), inv(r(i))], [[2, 3], [-1, -2, 1, 2], [1, 3, -4, -5], [-3, -7], [-6, -8]])
                    Ru = ncon([pr(j), c(A[j])], [[1, -1, -3, -4], [1, -2, -5]])
                    Rb = ncon([pr(j), pr(j), inv(r(j))], [[-3, -4, 1, -1], [1, -2, -6, -7], [-5, -8]])

                    Lbs = self.right_transfer(Lb, i, L-1)
                    Rds = self.left_transfer(Rd, 0, i)
                    Rus = self.left_transfer(Ru, 0, j)
                    Rbs = self.left_transfer(Rb, 0, j)

                    if not d*Din_1==Di:
                        for m, h in reversed(list(enumerate(H))):
                            if m > i:
                                # from gauge symmetry
                                if not i==j:
                                    continue
                                else:
                                    #AAHAA
                                    Am, Am_1 = self.data[m:m+2]
                                    Kr_ = ncon([Am_1@r(m+1), c(Am_1), Am, c(Am), h], [[2,4,1], [3,5,1], [6,-1,4], [7,-2,5], [7,3,6,2]])
                                    G += tr(Lbs(m)@Kr_, 0, -1, -2)

                            Am, Am_1 = self.data[m:m+2]

                            if m==i:
                                if j==i:
                                    # BAHBA
                                    G += ncon([l(m-1)]+[pr(m), inv(r(m))@Am_1@r(m+1)]+[h]+[pr(m), inv(r(m))@c(Am_1)],
                                              [[5, 6], [1, 5, -4, -5], [2, -6, 7], [1, 2, 3, 4], [-1, -2, 3, 6], [4, -3, 7]],
                                              [5, 6, 7, 1, 2, 3, 4])
                                elif j==i+1:
                                    # ABHBA
                                    G += ncon([l(m-1)]+[Am, pr(m+1)]+[h]+[pr(m), inv(r(m))@c(Am_1)],
                                             [[5, 6], [1, 6, 7], [2, 7, -4, -5], [1, 2, 3, 4], [-1, -2, 3, 5], [4, -3, -6]],
                                             [5, 6, 1, 3, 7, 2, 4])
                                else:
                                    # AAHBA
                                    O = ncon([l(m-1)@Am, Am_1]+[h]+[pr(m), inv(r(m))@c(Am_1)],
                                             [[3, 6, 5], [4, 5, -4], [1, 2, 3, 4], [-1, -2, 1, 6], [2, -3, -5]]) #(A)ud
                                    G += tensordot(O, Rus(m+2), [[-1, -2], [0, 1]])

                            elif m==i-1:
                                if j==i:
                                    # ABHAB
                                    G += ncon([l(m-1)@Am, pr(m+1)]+[h]+[c(Am), pr(m+1)]+[inv(r(m+1))],
                                              [[3, 5, 6], [4, 6, -4, -5], [1, 2, 3, 4], [1, 5, 7], [-1, -2, 2, 7], [-3, -6]],
                                              [1, 2, 3, 4, 5, 6, 7]) #AA

                                else:
                                    # AAHAB
                                    Q = ncon([l(m-1)@Am, Am_1]+[h]+[c(Am), pr(m+1)],
                                             [[3, 6, 5], [4, 5, -3], [1, 2, 3, 4], [1, 6, 7], [-1, -2, 2, 7]]) #(A)
                                    G += tensordot(Q, ncon([inv(r(m+1)), Rus(m+2)], [[-2, 1], [-1, 1, -3, -4, -5]]), [-1, 0])

                            elif m<i:
                                #AAHAA
                                C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
                                K = ncon([l(m-1)@c(Am), c(Am_1)]+[C],
                                         [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]])
                                if i==j:
                                    G += tensordot(K, Rbs(m+2), [[0, 1], [0, 1]])
                                else:
                                    G += tensordot(K, tensordot(Rds(m+2), ncon([inv(r(i)), Rus(i+1)], [[-2, 1], [-1, 1, -3, -4, -5]]), [-1, 0]),
                                                   [[0, 1], [0, 1]])
                            #AAHAA
                            C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA
                            K = ncon([c(l(m-1))@Am.conj(), Am_1.conj()]+[C],
                                     [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]])
                            # BAHBA
                            M = ncon([l(m-1)]+[pr(m), inv(r(m))@Am_1]+[h]+[pr(m), inv(r(m))@c(Am_1)]+[r(m+1)],
                                      [[5, 6], [1, 5, -4, -5], [2, -6, 7], [1, 2, 3, 4], [-1, -2, 3, 6], [4, -3, 8], [7, 8]])
                            # ABHBA
                            N = ncon([l(m-1)]+[Am, pr(m+1)]+[h]+[pr(m), r(m)@c(Am_1)],
                                     [[5, 6], [1, 6, 7], [2, 7, -4, -5], [1, 2, 3, 4], [-1, -2, 3, 5], [4, -3, -6]])
                            # AAHBA
                            O = ncon([l(m-1)@Am, Am_1]+[h]+[pr(m), inv(r(m))@c(Am_1)],
                                     [[3, 6, 5], [4, 5, -4], [1, 2, 3, 4], [-1, -2, 1, 6], [2, -3, -5]]) #(A)ud
                            # ABHAB
                            P = ncon([l(m-1)@Am, pr(m+1)]+[h]+[c(Am), pr(m+1)]+[r(m+1)],
                                     [[3, 5, 6], [4, 6, -4, -5], [1, 2, 3, 4], [1, 5, 7], [-1, -2, 2, 7], [-3, -6]]) #AA
                            # AAHAB
                            Q = ncon([l(m-1)@Am, Am_1]+[h]+[c(Am), pr(m+1)],
                                     [[3, 6, 5], [4, 5, -3], [1, 2, 3, 4], [1, 6, 7], [-1, -2, 2, 7]]) #(A)
                            assert allclose(norm(td(M, l(m-1)@c(Am), [[0, 1, 2], [0, 1, 2]])), 0)
                            assert allclose(norm(td(M, l(m-1)@Am, [[3, 4, 5], [0, 1, 2]])), 0)
                            assert allclose(norm(td(N, l(m-1)@c(Am), [[0, 1, 2], [0, 1, 2]])), 0)
                            assert allclose(norm(td(N, l(m)@Am_1, [[3, 4, 5], [0, 1, 2]])), 0)
                            assert allclose(norm(td(O, l(m-1)@c(Am), [[0, 1, 2], [0, 1, 2]])), 0)
                            assert allclose(norm(td(P, l(m)@c(Am_1), [[0, 1, 2], [0, 1, 2]])), 0)
                            assert allclose(norm(td(P, l(m)@Am_1, [[3, 4, 5], [0, 1, 2]])), 0)
                            assert allclose(norm(td(Q, l(m)@c(Am_1), [[0, 1, 2], [0, 1, 2]])), 0)

            elif fullH:
                i, j = (i_, j_)
                H = H.reshape([self.d, self.d]*self.L)
                links = self.links(True)
                bottom = [pr(m) if m==i else c(inv(r(m-1)))@a.conj() if m==i+1 else a.conj() for m, a in enumerate(A)]
                top = [c(pr(m)) if m==j else c(inv(r(m-1)))@a.conj() if m==j+1 else a.conj() for m, a in enumerate(A)]
                if i!=L-1 and j!=L-1:
                    links[i] = links[i][:2] + [-1, -2]
                    links[i+1][1] = -3
                    links[L+1+j] = links[L+1+j][:2] + [-4, -5]
                    links[L+1+j+1][1] = -6
                elif i!=L-1:
                    links[i] = links[i][:2] + [-1, -2]
                    links[i+1][1] = -3
                    links[L+1+j] = links[L+1+j][:2] + [-4, -5]
                    links[L-1][-1] = -6
                elif j!=L-1:
                    links[i] = links[i][:2] + [-1, -2]
                    links[2*L][-1] = -3
                    links[L+1+j] = links[L+1+j][:2] + [-4, -5]
                    links[L+1+j+1][1] = -6
                else:
                    links[i] = links[i][:2] + [-1, -2]
                    links[L+1+j] = links[L+1+j][:2] + [-4, -5]
                G = ncon(bottom+[H]+top, links)
                if i==L-1 and j==L-1:
                    G = ed(ed(G, -1), 2)
                return G


            if testing:
                # if testing is on, we compute the whole thing
                # with more checks and in a different way
                def gauge(G, i, j):
                    return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@c(vL(j)), inv(ch(r(i))), inv(ch(r(j)))], 
                                [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])
                assert allclose(G, gauge(G_, i, j))
                G = gauge(G_, i, j)
                G = c(tra(G, [3, 4, 5, 0, 1, 2])) if j_<i_ else G
                return G

            G_ = c(tra(G_, [2, 3, 0, 1])) if j_<i_ else G_
            self.F1_tot_ij_mem[str(i_)+str(j_)] = G_
            return G_

    def F2(self, i_, j_, H, envs=None, prs_vLs=None, fullH=False, testing=False, id=None):
        '''<d_id_j ψ|H|ψ>'''
        id = id if id is not None else uuid.uuid4().hex
        if self.id_ != id:
            self.id_ = id
            self.F2_ij_mem = {}
            self.F2_i_mem = {}
            self.F2_ij_mem_ = {}
            self.F2_i_mem_ = {}
            self.F2_tot_ij_mem = {}
        else:
            # read from cache: 
            # have we cached this tensor?
            if str(i_)+str(j_) in self.F2_tot_ij_mem:
                return self.F2_tot_ij_mem[str(i_)+str(j_)]

            # have we cached its transpose?
            if str(j_)+str(i_) in self.F2_tot_ij_mem:
                return tra(self.F2_tot_ij_mem[str(j_)+str(i_)], 
                           [3, 4, 5, 0, 1, 2])

        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs() if envs is None else envs
        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)] if prs_vLs is None else prs_vLs
        def pr(n): return prs_vLs[n][0]
        def vL(n): return prs_vLs[n][1]

        i, j = (j_, i_) if j_<i_ else (i_, j_)
        
        if i==j:
            if testing:
                G = 1j*zeros((*A[i].shape, *A[j].shape))

            gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
            gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1

            G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))
        elif not fullH:
            H = [h.reshape(2, 2, 2, 2) for h in H]
            d, Din_1, Di = self[i].shape

            if testing:
                G = 1j*zeros((*A[i].shape, *A[j].shape))
                _, Din_1, Di = self[i].shape
                if not d*Din_1==Di:
                    Rj = ncon([pr(j), A[j]], [[-3, -4, 1, -2], [1, -1, -5]])
                    Rjs = self.left_transfer(Rj, i, j)

                    Ri = ncon([pr(i), A[i]], [[-3, -4, 1, -2], [1, -1, -5]])
                    Ris = self.left_transfer(Ri, 0, i)

                    for m, h in reversed(list(enumerate(H))):
                        if m > i:
                            # from gauge symmetry
                            continue
                        Am, Am_1 = self.data[m:m+2]
                        C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA

                        if m==i:
                            if j==i+1:
                                G += ncon([l(m-1)@C, pr(i), dot(pr(j), inv(r(i)))], 
                                          [[1, 2, 3, -6], [-1, -2, 1, 3], [-4, -5, 2, -3]])
                            else:
                                L =  ncon([l(m-1)@C, pr(i), inv(r(i))@c(Am_1)], [[1, 2, 3, -4], [-1, -2, 1, 3], [2, -3, -5]]) # ud
                                G += ncon([L, Rjs(m+2)], [[-1, -2, -3, 1, 2], [1, 2, -4, -5, -6]])
                        elif m==i-1:
                            L = ncon([l(m-1)@C, c(Am), pr(i)], [[1, 2, 3, -3], [1, 3, 4], [-1, -2, 2, 4]])
                            G += ncon([L, ncon([Rjs(m+2), inv(r(i))], [[-1, 2, -3, -4, -5], [2, -2]])], [[-1, -2, 1], [1, -3, -4, -5, -6]])
                        else:
                            K = ncon([l(m-1)@Am.conj(), Am_1.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA

                            L = ncon([K, Ris(m+2)], [[1, 2], [1, 2, -1, -2, -3]])
                            G += ncon([ncon([L, Rjs(i+1)], [[-1, -2, 1], [1, -3, -4, -5, -6]]),
                                       inv(r(i))],
                                       [[-1, -2, 1, -4, -5, -6], [1, -3]])

            # new stuff
            gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
            gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1
            G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))

            icr, icl = [inv(ch(r(i))) for i in range(self.L)], [inv(ch(l(i))) for i in range(self.L)]
            cr, cl = [ch(r(i)) for i in range(self.L)], [ch(l(i)) for i in range(self.L)]
            def inv_ch_r(n): return icr[n]
            def inv_ch_l(n): return icl[n]
            def ch_r(n): return cr[n]
            def ch_l(n): return cl[n]

            if not d*Din_1==Di:
                if str(i)+str(j) not in self.F2_ij_mem:
                    # new stuff
                    Rj_ = ncon([inv_ch_l(j-1)@c(vL(j)), A[j]@ch_r(j)], [[1, -2, -3], [1, -1, -4]])
                    Rjs_ = self.left_transfer(Rj_, i, j)
                    self.F2_ij_mem_[str(i)+str(j)] = Rjs_
                else:
                    Rjs_ = self.F2_ij_mem_[str(i)+str(j)]

                if str(i) not in self.F2_i_mem:
                    # new stuff
                    Ri_ = ncon([inv_ch_l(i-1)@c(vL(i)), A[i]], [[1, -2, -3], [1, -1, -4]])
                    Ris_ = self.left_transfer(Ri_, 0, i)
                    self.F2_i_mem_[str(i)] = Ris_
                else:
                    Ris_ = self.F2_i_mem_[str(i)]

                for m, h in reversed(list(enumerate(H))):
                    if m > i:
                        # from gauge symmetry
                        continue
                    Am, Am_1 = self.data[m:m+2]
                    C = ncon([h]+[Am, Am_1], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]]) # HAA

                    if m==i:
                        if j==i+1:
                            G_+= ncon([ch_l(i-1)@C@ch_r(j), c(vL(i)), inv_ch_r(i)@inv_ch_l(j-1)@c(vL(j))],
                                      [[1, 2, 3, -4], [1, 3, -1], [2, -2, -3]])
                        else:
                            L_=  ncon([ch_l(i-1)@C, c(vL(i)), inv_ch_r(i)@c(Am_1)], [[3,4,1,-3], [3,1,-1], [4, -2, -4]])
                            G_+= tensordot(L_, Rjs_(m+2), [[2, 3], [0, 1]])
                    elif m==i-1:
                        L_ = ncon([l(m-1)@C, c(Am), inv_ch_l(i-1)@c(vL(i))], [[2, 4, 1, -2], [2, 1, 3], [4, 3, -1]])
                        G_+= tensordot(L_, ncon([inv_ch_r(i), Rjs_(m+2)], [[-2, 1], [-1, 1, -3, -4]]), [[-1], [0]])
                    else:
                        K = ncon([l(m-1)@Am.conj(), Am_1.conj()]+[C], [[1, 3, 4], [2, 4, -2], [1, 2, 3, -1]]) #AAHAA

                        L_ = tensordot(K, Ris_(m+2), [[0, 1], [0, 1]])
                        G_+= ncon([tensordot(L_, Rjs_(i+1), [1, 0]), inv_ch_r(i)], [[-1, 1, -3, -4], [1, -2]])

        elif fullH:
            H = H.reshape([self.d, self.d]*self.L)
            if i==j:
                G = zeros((*A[i].shape, *A[j].shape))
            else:
                links = self.links(True)
                if i==j-1:
                    bottom = [pr(i) if m==i else ncon([c(r(j-1)), pr(j)], [[-2, 1], [-1, 1, -3, -4]]) if m==j else a.conj() for m, a in enumerate(A)]
                else:
                    bottom = [pr(m) if (m==i or m==j) else a.conj() for m, a in enumerate(A)]
                    bottom = [c(inv(r(m-1)))@a.conj() if m==i+1 or m==j+1 else a for m, a in enumerate(bottom)]
                top = A
                links[i] = links[i][:2]+[-1, -2]
                links[i+1][1] = -3
                links[j] = links[j][:2]+[-4, -5]
                if j == L-1:
                    links[-1][-1] = -6
                else:
                    links[j+1][1] = -6
                G = ncon(bottom+[H]+top, links)

        def gauge(G, i, j):
            return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@vL(j), inv(ch(r(i))), inv(ch(r(j)))], 
                        [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])

        if testing:
            assert allclose(gauge(G_, i, j), G)

        G = gauge(G_, i, j)
        G = tra(G, [3, 4, 5, 0, 1, 2]) if j_<i_ else G
        self.F2_tot_ij_mem[str(i_)+str(j_)] = G
        return G

    def christoffel(self, i, j, k, envs=None, prs_vLs=None, id=None, testing=True, closed=(None, None, None)):
        """christoffel: return the christoffel symbol in basis c(A_i), c(A_j), A_k.
           Close indices i, j, k, with elements of closed tuple: i.e. (B_i, B_j, B_k).
           Will leave any indices marked none open :-<d_id_jψ|d_kψ>"""
        id = id if id is not None else uuid.uuid4().hex
        if self.id__ != id:
            self.id__ = id
            # initialize the memories 
            # we only don't try the cache on the first call from jac
            self.christ_tot_ij_mem = {}
        else:
            # read from cache: 
            # have we cached this tensor?
            if str(i)+str(j)+str(k) in self.christ_tot_ij_mem:
                return self.christ_tot_ij_mem[str(i)+str(j)+str(k)]

            ## have we cached its conjugate?
            if str(j)+str(i)+str(k) in self.christ_tot_ij_mem:
                return tra(self.christ_tot_ij_mem[str(j)+str(i)+str(k)], 
                           [3, 4, 5, 0, 1, 2])

        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs() if envs is None else envs
        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)] if prs_vLs is None else prs_vLs
        def pr(n): return prs_vLs[n][0]
        def vL(n): return prs_vLs[n][1]
        def Γ(i_, j_, k):
                """Γ: Christoffel symbol: does take advantage of gauge"""
                #j always greater than i (Γ symmetric in i, j)
                i, j = (j_, i_) if j_<i_ else (i_, j_)
                _, Din_1, Di = self[i].shape

                if j==i or i!=k or (d*Din_1==Di):
                    G = 1j*zeros((*A[i].shape, *A[j].shape, *A[k].shape))
                else:
                    R = ncon([pr(j), A[j]], [[-3, -4, 1, -2], [1, -1, -5]])
                    Rs = self.left_transfer(R, i, j)
                    G = ncon([pr(i), Rs(i+1), inv(r(i)), inv(r(i))], [[-1, -2, -7, -8], [1, 2, -4, -5, -6], [1, -9], [2, -3]])

                G = -tra(G, [3, 4, 5, 0, 1, 2, 6, 7, 8]) if j_<i_ else -G

                return G

        if any([c is not None for c in closed]):
            c_ind = [m for m, A in enumerate(closed) if A is not None]
            o_ind = [m for m, A in enumerate(closed) if A is None]
            links = [reduce(lambda x, y: x+y,
                           [list(array([-1, -2, -3])-3*(o_ind.index(n))) if n in o_ind else
                           [c_ind.index(n), c_ind.index(n)+3, c_ind.index(n)+6] for n in range(3)])]+\
                    [[n, n+3, n+6] for n in range(len(c_ind))]
            Γ_c = ncon([Γ(i, j, k)]+[closed[j] for j in c_ind], links)
        else:
            Γ_c = Γ(i, j, k)

        self.christ_tot_ij_mem[str(i)+str(j)+str(k)] = Γ_c
        return Γ_c

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

    def right_null_projector(self, n, r=None, get_vR=False, store_envs=False, vR=None):
        if r is None:
            _, r = self.get_envs(store_envs)
        if vR is None:
            R_ = sw(c(self[n])@r(n), 0, 1)
            R = R_.reshape(-1, self.d*R_.shape[-1])
            vR = sw(null(R).reshape(self.d, R_.shape[-1], -1), 1, 2)
        pr = ncon([inv(ch(r(n))), vR, c(vR), inv(ch(r(n)))], [[-2, 2], [-1, 1, 2], [-3, 1, 4], [-4, 4]])
        if get_vR:
            return pr, vR
        return pr

    def tangent_state(self, x, n, envs=None, vL=None):
        l , r = self.get_envs() if envs is None else envs
        _, vL = self.left_null_projector(n, l, get_vL=True) if vL is None else 1, vL
        return fMPS([inv(ch(l(n-1)))@vL@x if m==n else inv(ch(r(n)))@A if m==n+1 else A for m, A in enumerate(self.data)])

    def tangent_space_dims(self, l=None, get_vLs=False):
        l, _ = self.get_envs() if l is None else (l, None)
        vLs = [self.left_null_projector(n, l=l, get_vL=True)[1] for n in range(self.L)]
        shapes = [(vL.shape[-1], self.data[n+1].shape[1] if n+1<self.L else 1)
                  for vL, n in zip(vLs, range(self.L))]
        if get_vLs:
            return vLs, shapes
        else:
            return shapes

    def tangent_space_basis(self, type='eye'):
        """ return a tangent space basis
        """
        if type=='rand':
            Qs = [qr(randn(d1*d2, d1*d2)+1j*randn(d1*d2, d1*d2))[0]
                  for d1, d2 in self.tangent_space_dims() if d1*d2 != 0]
        elif type=='eye':
            Qs = [eye(d1*d2)+1j*0
                  for d1, d2 in self.tangent_space_dims() if d1*d2 != 0]
        def direct_sum(basis1, basis2):
            d1 = len(basis1[0])
            d2 = len(basis2[0])
            return [ct([b1, zeros(d2)]) for b1 in basis1]+\
                   [ct([zeros(d1), b2]) for b2 in basis2]
        return array(reduce(direct_sum, Qs))

    def extract_tangent_vector(self, dA):
        """extract_tangent_vector from dA:
           assume mps represents element of tangent space i.e.
           [B1...BN] <=> A1...Bn...AN + A1...Bn+1...AN+...
           return x1++x2++x3...++xN (concatenate + flatten)"""
        xs = []
        for n, shape in enumerate(self.tangent_space_dims()):
            if prod(shape) == 0:
                continue
            _, vL = self.left_null_projector(n, get_vL=True, store_envs=True)
            l, r = self.l, self.r
            x = ncon([c(vL), ch(l(n-1))@dA[n]@ch(r(n))], [[1, 2, -1], [1, 2, -2]])
            xs.append(x.reshape(-1))
        return ct(xs)

    def import_tangent_vector(self, v, xs=False):
        l, r = self.get_envs()
        vLs, shapes = self.tangent_space_dims(l, get_vLs=True)
        vLs = [vL for vL in vLs if prod(vL.shape)!=0]
        S = [shape for shape in shapes if prod(shape)!=0]
        nulls = len([shape for shape in shapes if prod(shape)==0])
        vs = chop(v, cs([prod(s) for s in S]))
        xs = [x.reshape(*shape) for x, shape in zip(vs, S)]
        l_, r_ = lambda n: l(n+nulls), lambda n: r(n+nulls)
        Bs = [zl(A) for A in self.data[:nulls]] + \
             [inv(ch(l_(n-1)))@vL@x@inv(ch(r_(n))) for n, (vL, x) in enumerate(zip(vLs, xs))]
        return fMPS(Bs)

class vfMPS(object):
    """vidal finite MPS
    lists of tuples (Gamma, Lambda) of numpy arrays"""
    def __init__(self,  data=None, d=None):
        """__init__

        :param data: matrices in form [(Gamma, Lambda)]
        :param d: local state space dimension
        """
        if data is not None and d is not None:
            self.L = len(data)
            self.d = d
            self.D = max([datum[0][0].shape for datum in data])

    def from_fMPS(self, fMPS):
        """from_fMPS: convert a normal form fMPS to vidal form

        :param fMPS: fMPS to convert
        """
        self.d = fMPS.d
        fMPS.right_canonicalise()
        data = []
        for L, A in zip(fMPS.Ls, fMPS.data):
            G = swapaxes(dot(inv(L), A), 0, 1)
            data.append((L, G))
        self.data = data
        return self

    def to_fMPS(self):
        """to_fMPS: return from vidal form to normal form"""
        return fMPS([swapaxes(dot(*x), 0, 1) for x in self.data], self.d)

class TestfMPS(unittest.TestCase):
    """TestfMPS"""

    def setUp(self):
        """setUp"""
        self.N = N = 5  # Number of MPSs to test
        #  min and max params for randint
        L_min, L_max = 6, 8
        d_min, d_max = 2, 4
        D_min, D_max = 5, 10
        ut_min, ut_max = 3, 7
        # N random MPSs
        self.rand_cases = [fMPS().random(randint(L_min, L_max),
                                         randint(d_min, D_min),
                                         randint(D_min, D_max))
                           for _ in range(N)]
        # N random MPSs, right canonicalised and truncated to random D
        self.right_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, D_min),
                            randint(D_min, D_max)).right_canonicalise(
                            randint(D_min, D_max))
                            for _ in range(N)]
        # N random MPSs, left canonicalised and truncated to random D
        self.left_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, D_min),
                            randint(D_min, D_max)).left_canonicalise(
                            randint(D_min, D_max))
                           for _ in range(N)]
        self.mixed_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, D_min),
                            randint(D_min, D_max)).mixed_canonicalise(
                            randint(ut_min, ut_max),
                            randint(D_min, D_max))
                            for _ in range(N)]

        # finite fixtures
        self.tens_0_2 = load('fixtures/mat2x2.npy')
        self.tens_0_3 = load('fixtures/mat3x3.npy')
        self.tens_0_4 = load('fixtures/mat4x4.npy')
        self.tens_0_5 = load('fixtures/mat5x5.npy')
        self.tens_0_6 = load('fixtures/mat6x6.npy')
        self.tens_0_7 = load('fixtures/mat7x7.npy')

        self.mps_0_2 = fMPS().left_from_state(self.tens_0_2)
        self.psi_0_2 = self.mps_0_2.recombine().reshape(-1)

        self.mps_0_3 = fMPS().left_from_state(self.tens_0_3)
        self.psi_0_3 = self.mps_0_3.recombine().reshape(-1)

        self.mps_0_4 = fMPS().left_from_state(self.tens_0_4)
        self.psi_0_4 = self.mps_0_4.recombine().reshape(-1)

        self.mps_0_5 = fMPS().left_from_state(self.tens_0_5)
        self.psi_0_5 = self.mps_0_5.recombine().reshape(-1)

        self.mps_0_6 = fMPS().left_from_state(self.tens_0_6)
        self.psi_0_6 = self.mps_0_6.recombine().reshape(-1)

        self.mps_0_7 = fMPS().left_from_state(self.tens_0_7)
        self.psi_0_7 = self.mps_0_7.recombine().reshape(-1)

    def test_energy_2(self):
        """test_energy_2: 2 spins: energy of random hamiltonian matches full H"""
        H2 = randn(4, 4)+1j*randn(4, 4)
        H2 = (H2+H2.conj().T)/2
        e_ed = self.psi_0_2.conj()@H2@self.psi_0_2
        e_mps = self.mps_0_2.energy(H2, fullH=True)
        self.assertTrue(isclose(e_ed, e_mps))

    def test_energy_3(self):
        """test_energy_3: 3 spins: energy of random hamiltonian matches full H"""
        H3 = randn(8, 8)+1j*randn(8, 8)
        H3 = (H3+H3.conj().T)/2
        e_ed = self.psi_0_3.conj()@H3@self.psi_0_3
        e_mps = self.mps_0_3.energy(H3, fullH=True)
        self.assertTrue(isclose(e_ed, e_mps))

    def test_energy_4(self):
        """test_energy_4: 4 spins: energy of random hamiltonian matches full H"""
        H4 = randn(16, 16)+1j*randn(16, 16)
        H4 = (H4+H4.conj().T)/2
        e_ed = self.psi_0_4.conj()@H4@self.psi_0_4
        e_mps = self.mps_0_4.energy(H4, fullH=True)
        self.assertTrue(isclose(e_ed, e_mps))

    def test_left_from_state(self):
        for _ in range(self.N):
            d = randint(2, 4)
            L = randint(4, 6)
            full = randn(*[d]*L) + 1j*randn(*[d]*L)
            full /= norm(full)
            case = fMPS().left_from_state(full)
            self.assertTrue(allclose(case.recombine(), full))

    def test_right_from_state(self):
        for _ in range(self.N):
            d = randint(2, 4)
            L = randint(4, 6)
            full = randn(*[d]*L) + 1j*randn(*[d]*L)
            full /= norm(full)
            case = fMPS().right_from_state(full)
            self.assertTrue(allclose(case.recombine(), -full) or
                            allclose(case.recombine(), full))

    def test_right_canonicalise(self):
        """test_right_canonicalise"""
        for case in self.right_cases:
            if not case.ok:
                print('\n')
                print('d: ', case.d, 'L: ', case.L, 'D: ', case.D)
                print('irc: ',
                      is_right_canonical(case.data, error=True),
                      case.is_right_canonical)
                print('irec: ', case.is_right_env_canonical)
                print('ifr: ', case.is_full_rank)
                print('tr1: ', case.has_trace_1)
                print('norm: ', case.norm())
                print('\n')
            self.assertTrue(case.ok)

    def test_left_canonicalise(self):
        """test_left_canonicalise"""
        for case in self.left_cases:
            if not case.ok:
                print('\n')
                print('d: ', case.d, 'L: ', case.L, 'D: ', case.D)
                print('ilc: ',
                      is_left_canonical(case.data, error=True),
                      case.is_left_canonical)
                print('ilec: ', case.is_left_env_canonical)
                print('fr: ', case.is_full_rank)
                print('tr1: ', case.has_trace_1)
                print('norm: ', case.norm())
                print('\n')
            self.assertTrue(case.ok)

    def test_left_canonicalise_norm(self):
        for case in self.rand_cases:
            I = []
            for _ in range(10):
                case.left_canonicalise()
                I.append(case.E(identity(case.d), 1))
            self.assertTrue(allclose(I, 1))

    def test_left_canonicalise_expectation_values(self):
        """EVs of spin operators on 3 random sites don't change after canonicalising 10 times"""
        for case in self.rand_cases:
            if case.d == 2:
                site1, site2, site3 = randint(0, case.L-1), randint(0, case.L-1), randint(0, case.L-1)
                S = [(Sx, site1), (Sy, site2), (Sz, site3)]
                I0 = [case.E(*opsite) for opsite in S]
                I = []
                for _ in range(10):
                    case.left_canonicalise()
                    I.append([case.E(*opsite) for opsite in S])
                for In in I:
                    self.assertTrue(allclose(I0, In))

    def test_right_canonicalise_expectation_values(self):
        """EVs of spin operators on 3 random sites don't change after canonicalising 10 times"""
        for case in self.rand_cases:
            if case.d == 2:
                site1, site2, site3 = randint(0, case.L-1), randint(0, case.L-1), randint(0, case.L-1)
                S = [(Sx, site1), (Sy, site2), (Sz, site3)]
                I0 = [case.E(*opsite) for opsite in S]
                I = []
                for _ in range(10):
                    case.right_canonicalise()
                    I.append([case.E(*opsite) for opsite in S])
                for In in I:
                    self.assertTrue(allclose(I0, In))

    def test_right_canonicalise_norm(self):
        """Norm doesn't change after canonicalising ten times"""
        for case in self.rand_cases:
            I = []
            for _ in range(10):
                case.right_canonicalise()
                I.append(case.E(identity(case.d), 1))
            self.assertTrue(allclose(I, 1))

    def test_mixed_canonicalise(self):
        """test_mixed_canonicalise"""
        for case in self.mixed_cases:
            if not case.ok:
                print('\n')
                print('d: ', case.d, 'L: ', case.L, 'D: ', case.D)
                print('irc: ', is_right_canonical(case.data[case.oc+1:],
                                                  error=True))
                print('ilc: ', is_left_canonical(case.data[:case.oc],
                                                 error=True))
                print('irec: ', case.is_right_env_canonical)
                print('ifr: ', case.is_full_rank)
                print('tr1: ', case.has_trace_1)
                print('norm: ', case.norm())
                print('\n')
            self.assertTrue(case.ok)

    def test_self_D_is_correct(self):
        for case in self.left_cases + self.right_cases:
            self.assertTrue(case.D >= max([max(x[0].shape) for x in case.data]))

    def test_apply(self):
        cases = [fMPS().random(5, 2, 3).left_canonicalise() for _ in range(5)]
        for case in cases:
            self.assertTrue(case.apply((eye(2), 0))==case)
            self.assertTrue(case.apply((eye(2), 1))==case)
            self.assertTrue(case.apply((eye(2), 2))==case)
            self.assertTrue(case.apply((eye(2), 3))==case)
            self.assertTrue(case.apply((eye(2), 4))==case)
            self.assertTrue(case.apply((Sx, 0)).apply((Sx, 0))==case)
            self.assertTrue(case.apply((Sy, 0)).apply((Sy, 0))==case)
            self.assertTrue(case.apply((Sz, 0)).apply((Sz, 0))==case)
            self.assertTrue(case.copy().apply((Sz, 0))!=case)

    def test_left_norms(self):
        """test_left_norms"""
        for case in self.left_cases:
            if not isclose(case.norm(), 1):
                case.update_properties()
                print('left')
                print(case.norm())
                print('L: ', case.L)
                print('D: ', case.D)
                print('d: ', case.d)
                print(case.structure())
                print(case.is_left_canonical)
                try:
                    print('oc: ', case.oc)
                except:
                    print('no oc')
            self.assertTrue(isclose(case.norm(), 1))

    def test_right_norms(self):
        """test_right_norms"""
        for case in self.right_cases:
            if not isclose(case.norm(), 1):
                print('right')
                print(case.norm())
                print('L: ', case.L)
                print('D: ', case.D)
                print('d: ', case.d)
                print(case.structure())
                try:
                    print('oc: ', case.oc)
                except:
                    print('no oc')
            self.assertTrue(isclose(case.norm(), 1))

    def test_serialize_deserialize(self):
        mps = self.mps_0_4
        mps_ = fMPS().deserialize(mps.serialize(), mps.L, mps.d, mps.D)
        self.assertTrue(mps==mps_)

    def test_store_load(self):
        mps = self.mps_0_2
        mps.store('x')
        mps_ = fMPS().load('x.npy')
        self.assertTrue(mps==mps_)

        mps = self.mps_0_3
        mps.store('x')
        mps_ = fMPS().load('x.npy')
        self.assertTrue(mps==mps_)

        mps = self.mps_0_4
        mps.store('x')
        mps_ = fMPS().load('x.npy')
        self.assertTrue(mps==mps_)

    def test_serialize_deserialize_real(self):
        mps = self.mps_0_4
        mps_ = fMPS().deserialize(mps.serialize(True), mps.L, mps.d, mps.D, True)
        self.assertTrue(mps==mps_)

    def test_import_extract_tangent_vector(self):
        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_4
        H = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22]
        dA = mps.dA_dt(H)
        dA_ = mps.import_tangent_vector(mps.extract_tangent_vector(dA))
        self.assertTrue(dA==dA_)

    def test_local_hamiltonians_2(self):
        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_2
        listH = [Sz12@Sz22+Sx12+Sx22]
        fullH = listH[0]
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_local_hamiltonians_3(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 3)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 3)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 3)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_3
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22]
        fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_local_hamiltonians_4(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_4
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22]
        fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_local_hamiltonians_5(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 5)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 5)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 5)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 5)
        Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 5)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_5
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx12+Sx22, Sz12@Sz12+Sx22]
        fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_local_hamiltonians_6(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 6)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 6)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 6)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 6)
        Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 6)
        Sx6, Sy6, Sz6 = N_body_spins(0.5, 6, 6)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_6
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx12+Sx22, Sz12@Sz12+Sx12+Sx22, Sz12@Sz12+Sx22]
        fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_local_hamiltonians_7(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 7)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 7)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 7)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 7)
        Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 7)
        Sx6, Sy6, Sz6 = N_body_spins(0.5, 6, 7)
        Sx7, Sy7, Sz7 = N_body_spins(0.5, 7, 7)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_7
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22]
        fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
        self.assertTrue(mps.dA_dt(listH, fullH=False)==mps.dA_dt(fullH, fullH=True))

    def test_left_transfer(self):
        mps = self.mps_0_4.left_canonicalise()
        l, r = mps.get_envs()
        L = mps.L
        rs = mps.left_transfer(r(L-1), 1, L)
        for i in range(L):
            self.assertTrue(allclose(rs(i+1), r(i)))
            self.assertTrue(allclose(mps.left_transfer(r(L-1), i+1, L, False), r(i)))

    def test_right_transfer(self):
        mps = self.mps_0_4.right_canonicalise()
        l, r = mps.get_envs()
        L = mps.L
        ls = mps.right_transfer(l(0), 0, L-1)
        for i in range(L):
            self.assertTrue(allclose(ls(i+1), l(i)))

    def test_local_recombine(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        listH4 = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22]
        comH4 = sum([n_body(a, i, len(listH4), d=2) for i, a in enumerate(listH4)], axis=0)
        fullH4 = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4+Sx1+Sx2+Sx3+Sx4
        self.assertTrue(allclose(fullH4, comH4))

    def test_local_energy(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_4
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22]
        fullH = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4+Sx1+Sx2+Sx3+Sx4
        self.assertTrue(isclose(mps.energy(listH, fullH=False), mps.energy(fullH, fullH=True)))

    def test_F2_F1(self):
        '''<d_id_j ψ|H|ψ>, <d_iψ|H|d_jψ>'''
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 6)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 6)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 6)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 6)
        Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 6)
        Sx6, Sy6, Sz6 = N_body_spins(0.5, 6, 6)

        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_6.left_canonicalise()
        listH = [Sz12@Sz22+Sx12, Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22+Sx12+Sx22, Sz12@Sz22+Sz12+Sx22, Sz12@Sz22+Sx22]
        eyeH = [(1/(mps.L-1))*eye(4) for _ in range(5)]
        fullH = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4+Sz4@Sz5+Sz5@Sz6+Sx1+Sx2+Sx3+Sx4+Sx5+Sx6

        for i, j in product(range(mps.L), range(mps.L)):
            ## F2 = <d_id_j ψ|H|ψ>
            # zero for H = I
            self.assertTrue(allclose(mps.F2(i, j, eyeH, testing=True), 0))

            # Test gauge projectors are in the right place
            mps.right_canonicalise()
            l, r = mps.get_envs()
            z1 = ncon([mps.F2(i, j, listH, testing=True), l(i-1)@c(mps[i])], [[1, 2, 3, -1, -2, -3], [1, 2, 3]])
            z2 = ncon([mps.F2(i, j, listH, testing=True), l(j-1)@c(mps[j])], [[-1, -2, -3, 1, 2, 3], [1, 2, 3]])
            self.assertTrue(allclose(z1, 0))
            self.assertTrue(allclose(z2, 0))

            mps.left_canonicalise()
            l, r = mps.get_envs()
            z1 = ncon([mps.F2(i, j, listH, testing=True), l(i-1)@c(mps[i])], [[1, 2, 3, -1, -2, -3], [1, 2, 3]])
            z2 = ncon([mps.F2(i, j, listH, testing=True), l(j-1)@c(mps[j])], [[-1, -2, -3, 1, 2, 3], [1, 2, 3]])
            self.assertTrue(allclose(z1, 0))
            self.assertTrue(allclose(z2, 0))

            ## F1 = <d_iψ|H|d_jψ>
            # For H = I: should be equal to δ_{ij} pr(i)
            if i!=j:
                self.assertTrue(allclose(mps.F1(i, j, eyeH, testing=True), 0))
            if i==j:
                b = mps.F1(i, j, eyeH, testing=True)
                a = ncon([mps.left_null_projector(i), inv(r(i))], [[-1, -2, -4, -5], [-3, -6]])
                self.assertTrue(allclose(a, b))

            # Test gauge projectors are in the right place
            mps.left_canonicalise()
            l, r = mps.get_envs()
            z1 = td(mps.F1(i, j, listH, testing=True), l(i-1)@c(mps[i]), [[0, 1, 2], [0, 1, 2]])
            z1 = td(mps.F1(i, j, listH, testing=True), l(j-1)@mps[j], [[3, 4, 5], [0, 1, 2]])
            self.assertTrue(allclose(z1, 0))
            self.assertTrue(allclose(z2, 0))

            mps.right_canonicalise()
            l, r = mps.get_envs()
            z1 = td(mps.F1(i, j, listH, testing=True), l(i-1)@c(mps[i]), [[0, 1, 2], [0, 1, 2]])
            z1 = td(mps.F1(i, j, listH, testing=True), l(j-1)@mps[j], [[3, 4, 5], [0, 1, 2]])
            self.assertTrue(allclose(z1, 0))
            self.assertTrue(allclose(z2, 0))

            # TODO: fullH fails gauge projectors test (listH doesn't):
            # TODO: fullH different from listH:

    def test_F2_F1_christoffel(self):
        '''-1j<d_id_j ψ|H|ψ>=<d_id_j ψ|Ad_j|d_jψ> with no truncation'''
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_3
        H = [Sz1@Sz2+Sz1, Sz1@Sz2+Sz1+Sz2]
        dA_dt = mps.dA_dt(H)
        l, r = mps.l, mps.r
        i, j = 1, 2
        F2 = -1j*mps.F2(i, j, H, testing=True)
        Γ2 = td(mps.christoffel(i, j, i), l(i-1)@dA_dt[i]@r(i), [[-3, -2, -1], [0, 1, 2]])
        self.assertTrue(allclose(F2+Γ2, 0))

        mps = self.mps_0_4
        H = [Sz1@Sz2+Sz1, Sz1@Sz2+Sz1+Sz2, Sz1@Sz2+Sz2]
        dA_dt = mps.dA_dt(H)
        l, r = mps.l, mps.r
        i, j = 2, 3
        F2 = -1j*mps.F2(i, j, H, testing=True)
        Γ2 = td(mps.christoffel(i, j, i), l(i-1)@dA_dt[i]@r(i), [[-3, -2, -1], [0, 1, 2]])
        self.assertTrue(allclose(F2+Γ2, 0))

        mps = self.mps_0_5
        H = [Sz1@Sz2+Sz1, Sz1@Sz2+Sz1+Sz2, Sz1@Sz2+Sz1+Sz2, Sz1@Sz2+Sz2]
        dA_dt = mps.dA_dt(H)
        l, r = mps.l, mps.r
        for i, j in [(2, 3), (2, 4), (3, 2), (4, 2)]:
            F2 = -1j*mps.F2(i, j, H, testing=True)
            Γ2 = td(mps.christoffel(i, j, min(i, j)), l(min(i, j)-1)@dA_dt[min(i, j)]@r(min(i, j)), [[-3, -2, -1], [0, 1, 2]])
            self.assertTrue(allclose(F2+Γ2, 0))

    def test_christoffel(self):
        mps = self.mps_0_6.left_canonicalise()
        ijks = ((4, 5, 4), (3, 5, 3), (3, 4, 3)) # all non zero indexes (for full rank)
        for i, j, k in ijks:
            # Gauge projectors are in the right place
            mps.left_canonicalise()
            l, r = mps.get_envs()
            self.assertTrue(not allclose(mps.christoffel(i, j, k), 0))
            i_true=allclose(mps.christoffel(i, j, k, closed=(l(i-1)@c(mps[i]), None, None)), 0)
            i_false=allclose(mps.christoffel(i, j, k, closed=(l(i-1)@mps[i], None, None)), 0)
            j_true=allclose(mps.christoffel(i, j, k, closed=(None, l(j-1)@c(mps[j]), None)), 0)
            j_false=allclose(mps.christoffel(i, j, k, closed=(None, l(j-1)@mps[j], None)), 0)
            k_true=allclose(mps.christoffel(i, j, k, closed=(None, None, l(k-1)@mps[k])), 0)
            k_false=allclose(mps.christoffel(i, j, k, closed=(None, None, l(k-1)@c(mps[k]))), 0)
            self.assertTrue(i_true)
            self.assertTrue(j_true)
            self.assertTrue(k_true)
            self.assertTrue(not i_false)
            self.assertTrue(not j_false)
            self.assertTrue(not k_false)

            mps.right_canonicalise()
            l, r = mps.get_envs()
            self.assertTrue(not allclose(mps.christoffel(i, j, k), 0))
            i_true=allclose(mps.christoffel(i, j, k, closed=(l(i-1)@c(mps[i]), None, None)), 0)
            i_false=allclose(mps.christoffel(i, j, k, closed=(l(i-1)@mps[i], None, None)), 0)
            j_true=allclose(mps.christoffel(i, j, k, closed=(None, l(j-1)@c(mps[j]), None)), 0)
            j_false=allclose(mps.christoffel(i, j, k, closed=(None, l(j-1)@mps[j], None)), 0)
            k_true=allclose(mps.christoffel(i, j, k, closed=(None, None, l(k-1)@mps[k])), 0)
            k_false=allclose(mps.christoffel(i, j, k, closed=(None, None, l(k-1)@c(mps[k]))), 0)
            self.assertTrue(i_true)
            self.assertTrue(j_true)
            self.assertTrue(k_true)
            self.assertTrue(not i_false)
            self.assertTrue(not j_false)
            self.assertTrue(not k_false)

            # symmetric in i, j
            self.assertTrue(allclose(mps.christoffel(i, j, k, closed=(c(mps[i]), c(mps[j]), mps[k])),
                                     mps.christoffel(j, i, k, closed=(c(mps[j]), c(mps[i]), mps[k]))  ))

            self.assertTrue(allclose(tra(mps.christoffel(i, j, k), [3, 4, 5, 0, 1, 2, 6, 7, 8]), mps.christoffel(j, i, k)))


        ijks = ((1, 2, 1),)
        for i, j, k in ijks:
            # Christoffel symbols that are zero for untruncated become not zero after truncation
            self.assertTrue(allclose(mps.christoffel(i, j, k), 0))
            mps.left_canonicalise(2)
            self.assertTrue(not allclose(mps.christoffel(i, j, k), 0))

    def test_ddA_dt(self):
        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_2
        eyeH = [eye(4)]
        dt = 0.1
        Z = [randn(3)+1j*randn(3) for _ in range(10)]

        for z in Z:
            self.assertTrue(allclose(mps.ddA_dt(z, eyeH), -1j*z))

    def test_null_projectors(self):
        mps = self.mps_0_4.right_canonicalise()
        for n in range(3):
            _, vR = mps.right_null_projector(n, get_vR=True, store_envs=True)
            _, vL = mps.left_null_projector(n, get_vL=True)
            self.assertTrue(allclose(ncon([mps[n]@ch(mps.r(n)), vR], [[1, -1, 3], [1, -2, 3]]), 0))
            self.assertTrue(allclose(ncon([ch(mps.l(n-1))@mps[n], vL], [[1, 3, -1], [1, 3, -2]]), 0))

    def test_dynamical_expand(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

        # need to expand with 2 site heisenberg hamiltonian
        from mps_examples import comp_z
        H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]

        # up down: must project
        mps = comp_z(2).left_canonicalise()
        self.assertTrue(not allclose(mps.projection_error(H, 0.1), 0))

        # up up: no projection
        mps = comp_z(1).left_canonicalise()
        self.assertTrue(allclose(mps.projection_error(H, 0.1), 0))

        # no need to expand with local hamiltonian
        for D in [1, 2, 3]:
            H = [Sx1+Sx2+Sy1+Sy2+Sz1+Sz2]*3
            mps = self.mps_0_4.left_canonicalise(D)

            self.assertTrue(allclose(mps.projection_error(H, 0.1), 0))
            pre_struc = mps.structure()
            mps.dynamical_expand(H, 0.1, 2*D)
            post_struc = mps.structure()
            self.assertTrue(pre_struc==post_struc)

        # need to expand with 4 site heisenberg hamiltonian
        H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]*3
        D = 1
        mps = self.mps_0_4.left_canonicalise(D)

        pre_struc = mps.structure()
        mps.dynamical_expand(H, 0.1, 2*D)
        post_struc = mps.structure()
        self.assertTrue(pre_struc!=post_struc)

    def test_grow(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_4.left_canonicalise(1)
        H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]*3
        for D in [1, 2, 3, 4]:
            A = mps.copy().grow(H, 0.1, D)
            self.assertTrue(A.D == D)
        A = mps.copy().grow(H, 0.1, 5)
        self.assertTrue(A.D == 4)

    def test_jac_eye(self):
        mps = self.mps_0_2
        H = [eye(4)]
        J1, J2 = mps.jac(H, True, False)
        self.assertTrue(allclose(J1, -1j*eye(3)))
        self.assertTrue(allclose(J2, 0))
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J, kron(1j*Sy, eye(3))))

        mps = self.mps_0_3
        H = [eye(4)/2, eye(4)/2]
        J1, J2 = mps.jac(H, True, False)
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J1, -1j*eye(7)))
        self.assertTrue(allclose(J2, 0))
        self.assertTrue(allclose(J, kron(1j*Sy, eye(7))))

        mps = self.mps_0_4
        H = [eye(4)/3, eye(4)/3, eye(4)/3]
        J1, J2 = mps.jac(H, True, False)
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J1, -1j*eye(15)))
        self.assertTrue(allclose(J2, 0))
        self.assertTrue(allclose(J, kron(1j*Sy, eye(15))))

    def test_jac_no_projection(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_3.left_canonicalise()
        H = [Sz1@Sz2+Sx1, Sz1@Sz2+Sx1+Sx2]
        J1, J2 = mps.jac(H, True, False)
        self.assertTrue(allclose(J1+J1.conj().T, 0))

        mps = fMPS().random(5, 2, 10).left_canonicalise()
        N = 5
        for _ in range(N):
            H = [randn(4, 4)+1j*randn(4, 4) for _ in range(4)]
            H = [h+h.conj().T for h in H]
            J1, J2 = mps.jac(H, True, False)
            self.assertTrue(allclose(J1,-J1.conj().T))
            self.assertTrue(allclose(J2, 0))
            J = mps.jac(H, True, True)
            self.assertTrue(allclose(J+J.T, 0))
            mps = (mps+mps.dA_dt(H)*0.1).left_canonicalise()

    def test_expand(self):
        mps = fMPS().random(3, 2, 1)
        self.assertTrue(mps.structure()==[(1, 1), (1, 1), (1, 1)])
        ls = mps.Es([Sx, Sy, Sz], 0)

        mps.expand(2)
        l, r = mps.get_envs()

        self.assertTrue(allclose(ls, mps.Es([Sx, Sy, Sz], 0)))
        self.assertTrue(mps.structure()==[(1, 2), (2, 2), (2, 1)])

class TestvfMPS(unittest.TestCase):
    """TestvfMPS"""

    def setUp(self):
        """setUp"""
        N = 20  # Number of MPSs to test
        #  min and max params for randint
        L_min, L_max = 9, 20
        d_min, d_max = 2, 5
        D_min, D_max = 5, 40
        # N random MPSs
        self.cases = [fMPS().random(randint(L_min, L_max),
                                    randint(d_min, d_max),
                                    randint(D_min, D_max))
                      for _ in range(N)]
        # N random MPSs right canonicalised with truncation
        self.right_cases = [fMPS().random(
                                randint(L_min, L_max),
                                randint(d_min, d_max),
                                randint(D_min, D_max)).right_canonicalise(
                                randint(D_min, D_max))
                            for _ in range(N)]

    def test_vidal_to_and_from_fMPS(self):
        """test_to_from_fMPS"""
        other_cases = [vfMPS().from_fMPS(case).to_fMPS() for case in self.cases]
        self.assertTrue(array([fMPS1 == fMPS2
                               for fMPS1, fMPS2 in zip(self.cases,
                                                       other_cases)]).all())

if __name__ == '__main__':
    unittest.main(verbosity=2)
