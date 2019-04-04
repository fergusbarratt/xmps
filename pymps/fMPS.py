"""fMPS: Finite length matrix product states"""
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import unittest

from numpy.random import rand, randint, randn
from numpy.linalg import svd, inv, norm, cholesky as ch, qr
from numpy.linalg import eig, eigvalsh

from numpy import array, concatenate, diag, dot, allclose, isclose, swapaxes as sw
from numpy import identity, swapaxes, trace, tensordot, sum, prod, ones
from numpy import real as re, stack as st, concatenate as ct, zeros, empty
from numpy import split as chop, ones_like, save, load, zeros_like as zl
from numpy import eye, cumsum as cs, sqrt, expand_dims as ed, imag as im
from numpy import transpose as tra, trace as tr, tensordot as td, kron
from numpy import mean, sign, angle, unwrap, exp, diff, pi, squeeze as sq
from numpy import round, flipud, cos, sin, exp, arctan2, arccos, sign

from scipy.linalg import null_space as null, orth, expm#, sqrtm as ch
from scipy.linalg import polar
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .tests import is_right_canonical, is_right_env_canonical, is_full_rank
from .tests import is_left_canonical, is_left_env_canonical, has_trace_1

from .tensor import H as cT, truncate_A, truncate_B, diagonalise, rank, mps_pad
from .tensor import C as c, lanczos_expm, tr_svd, T
from .tensor import rdot, ldot, structure
from .left_transfer import lt as lt_

from .spin import n_body, N_body_spins, spins
from copy import deepcopy, copy
from functools import reduce
from itertools import product
import cProfile
from time import time
import uuid

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz

from .ncon import ncon as ncon
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

    def __str__(self):
        return 'fMPS: L={}, d={}, D={}'.format(self.L, self.d, self.D)

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

    def random_with_energy_E(self, E, H, L, d, D, tol=1e-10, maxiters=100):
        """random_with_energy_E: converges a random state to energy E within tol (quite slow)

        :param E: energy
        :param H: hamiltonian
        :param L: Length of chain
        :param d: local state space dimension
        :param D: bond dimension"""
        self = self.random(L, d, D).left_canonicalise()

        dt = 0.1
        iters = 0
        while abs(self.energy(H)-E)>tol:
            if iters>maxiters:
                raise Exception("Too many iterations")
                break
            iters+=1
            dE = self.energy(H)-E
            ddE = 10
            if dE < 0:
                while (self.energy(H)-E) < 0:
                    if ddE<1e-5 and abs(dE)>1e-2:
                        raise Exception("local minimum")
                    self.data = (self+1j*dt*self.dA_dt(H)).left_canonicalise().data

                    #print('up')
                    ddE = abs((self.energy(H)-E)-dE)
                    #print('ddE: ', ddE)
                    dE = self.energy(H)-E
                    #print('dE: ', dE)
            else:
                while (self.energy(H)-E) > 0:
                    if ddE<1e-5 and abs(dE)>1e-2:
                        raise Exception("local minimum")
                    self.data = (self-1j*dt*self.dA_dt(H)).left_canonicalise().data

                    #print('down')
                    ddE = abs((self.energy(H)-E)-dE)
                    #print('ddE: ', ddE)
                    dE = self.energy(H)-E
                    #print('dE: ', dE)

            dt = dt/3
        return self

    def random(self, L, d, D):
        """__init__

        :param L: Length
        :param d: local state space dimension
        :param D: bond dimension
        generate a random fMPS
        """
        self.L = L
        self.d = d
        if D == 1:
            self.D = 1
            # random product states
            V = randn(L, 3)
            V = V/ed(norm(V, axis=-1), -1)
            def to_spherical(rs):
                sphers = []
                for r in rs:
                    [x, y, z] = r
                    sphers.append([norm(r), arctan2(y,z), arccos(z/norm(r))])
                return array(sphers)
            V = to_spherical(V)
            self.data = []
            for _, th, f in V:
                self.data.append(ed(ed(array([cos(th/2), exp(1j*f)*sin(th/2)]), -1), -1))
            return self

        MPS = [randn(*((d,) + shape)) + 1j*randn(*((d,) + shape))
                for shape in self.create_structure(L, d, D)]
        self.D = max([max(shape[1:]) for shape in self.create_structure(L, d, D)])
        self.data = MPS
        return self

    def create_structure(self, L, d, D, phys=False):
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
        if phys:
            return [(self.d, *st) for st in structure]
        else:
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
                ls.append(diag(diag(sum(re(cT(mps[n]) @ ls[-1] @ mps[n]), axis=0))))
            return lambda n: ls[n+1]

        def get_r(mps):
            rs = [array([[1]])]
            for n in range(len(mps))[::-1]:
                rs.append(diag(diag(sum(re(mps[n] @ rs[-1] @ cT(mps[n])), axis=0))))
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
                h = h.reshape(*[self.d]*4)
                C = ncon([h]+self.data[m:m+2], [[-1, -2, 1, 2], [1, -3, 3], [2, 3, -4]])
                e.append(ncon([c(l(m-1))@self.data[m].conj(), self.data[m+1].conj()@c(r(m+1))]+[C], [[1, 3, 4], [2, 4, 5], [1, 2, 3, 5]]))
            return re(sum(e))
        else:
            return self.E_L(H)

    def serialize(self, real=False):
        """serialize: return a vector with mps data in it"""
        vec = ct([a.reshape(-1) for a in self])
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

    def projection_error(self, H, dt):
        """projection_error: error in projection to mps manifold

        :param H: hamiltonian
        :param dt: time step
        """
        L, d, A, D = self.L, self.d, self.data, self.D
        dA_dt = self.dA_dt(H, True)
        l, r = self.l, self.r
        def vR(n): return self.right_null_projector(n, r, True)[1]

        H = [h.reshape(*[self.d]*4) for h in H]
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

    def expand(self, D_):
        """expand - just pad with zeros"""
        from numpy import pad
        L, d, D = self.L, self.d, self.D
        for m, (sh, sh_) in enumerate(zip(self.structure(), self.create_structure(L, d, D_))):
            self[m] = pad(self[m], list(zip([0, 0, 0], (0, *tuple(array(sh_)-array(sh))))), 'constant')
        self.D = D_
        return self.left_canonicalise(minD=False)

    def should_expand(self, H, dt, threshold):
        """should_expand the manifold?
        """
        return self.projection_error(H, dt) > threshold

    def dynamical_expand(self, H, dt, D_, threshold=1e-8):
        """dynamical_expand: expand bond dimension to D_ during timestep dt with H

        :param H: hamiltonian
        :param dt: timestep
        :param D_: new bond dimension
        :param threshold: by default will expand if no need to, else add threshold
        """
        L, d, A, D = self.L, self.d, self.data, self.D
        def vR(n): return self.right_null_projector(n, r, True)[1]

        H = [h.reshape(*[self.d]*4) for h in H]
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

    def dA_dt(self, H, store_energy=False, fullH=False, prs_vLs=None):
        """dA_dt: Finds A_dot (from TDVP) [B(n) for n in range(n)], energy. Uses inverses.
        Indexing is A[0], A[1]...A[L-1]

        :param self: matrix product state @ current time
        :param H: Hamiltonian
        """
        self.dA_cache = {}
        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs(True)
        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)] if prs_vLs is None else prs_vLs
        def pr(n): return prs_vLs[n][0]
        def vL(n): return prs_vLs[n][1]

        if not fullH:
            def B(i, H=H):
                e = []
                B = -1j*zl(A[i])
                _, Dn, Dn_1 = A[i].shape
                H = [h.reshape(d, d, d, d) for h in H]

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

    def match_gauge_to(self, mps):
        """match gauge with another matrix product state
           canonical matrix product state is unique up to diagonal matrices
           in the svds, and unitary freedom in the tangent spaces.
           match from one timestep to the next makes trajectories continuous.
        """
        for i in reversed(range(self.L)):
            S = sum(self[i]@cT(mps[i]), axis=0)
            U, P = polar(S)
            self[i] = cT(U)@self[i]

            if 0 < i:
                self[i-1] = self[i-1]@U
        return self

    def jac(self, H,
            as_matrix=True,
            real_matrix=True,
            fix_vLs=True):
        """jac: calculate the jacobian of the current mps
        """
        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs()

        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)]
        prs = [x[0] for x in prs_vLs]
        vLs = [x[1] for x in prs_vLs]

        self.new_vL = vLs
        if fix_vLs:
            if hasattr(self, 'old_vL'):
                self.v = array([])
                vLs = []
                for i in range(self.L):
                    if prod(A[i].shape[:-1])!=A[i].shape[-1]:
                        S = sum(cT(self.new_vL[i])@self.old_vL[i], axis=0)
                        U, P = polar(S)
                        vLs.append(self.new_vL[i]@U)
                        self.v = ct([self.v, vLs[i].real.reshape(-1), vLs[i].imag.reshape(-1)])
                    else:
                        vLs.append(self.new_vL[i])
                self.new_vL = vLs
            prs = [self.left_null_projector(n, l, vL=vLs[n]) for n in range(self.L)]

        prs_vLs = list(zip(prs, vLs))
        def vL(i): return vLs[i]
        dA_dt = self.dA_dt(H, prs_vLs=prs_vLs)
        def inv_diag(mat):
            return diag(1/diag(mat))
        def ch_diag(mat):
            return diag(sqrt(diag(mat)))
        cr, cl = [ch_diag(r(i)) for i in range(self.L)], [ch_diag(l(i)) for i in range(self.L)]
        icr, icl = [inv_diag(cr[i]) for i in range(self.L)], [inv_diag(cl[i]) for i in range(self.L)]
        inv_envs= (icr, icl, cr, cl)

        # Get tensors
        ## unitary rotations: -<d_iψ|(d_t |d_kψ> +iH|d_kψ>) (dA_k)
        #-<d_iψ|d_kd_jψ> dA_j/dt (dA_k) (range(k+1, L))
        #def Γ1(i, k): return sum([td(c(self.christoffel(k, j, i, envs=(l, r))), l(j-1)@dA_dt[j]@r(j), [[3, 4, 5], [0, 1, 2]]) for j in range(L)], axis=0)
        #-i<d_iψ|H|d_kψ> (dA_k)
        id = uuid.uuid4().hex # for memoization
        def F1(i, k): return  -1j*self.F1(i, k, H, envs=(l, r), inv_envs=inv_envs, prs_vLs=prs_vLs, id=id)

        ## non unitary (from projection): -<d_id_kψ|(d_t |ψ> +iH|ψ>) (dA_k*) (should be zero for no projection)
        #-<d_id_kψ|d_jψ> dA_j/dt (dA_k*)
        def Γ2(i, k): return self.christoffel(i, k, min(i, k), envs=(l, r), prs_vLs=prs_vLs, closed=(None, None, l(min(i, k)-1)@dA_dt[min(i, k)]@r(min(i, k))))

        #-i<d_id_k ψ|H|ψ> (dA_k*)
        def F2(i, k): return -1j*self.F2(i, k, H, envs=(l, r), inv_envs=inv_envs, prs_vLs=prs_vLs, id=id)

        sh = self.tangent_space_dims(l, vLs=vLs)
        nulls = len([1 for (a, b) in sh if a==0 or b==0])
        shapes = list(cs([prod([a, b]) for (a, b) in sh if a!=0 and a!=0]))
        DD = shapes[-1]
        def ind(i):
            slices = [slice(a[0], a[1], 1)
                      for a in [([0]+shapes)[i:i+2] for i in range(len(shapes))]]
            return slices[i]

        J1_ = -1j*zeros((DD, DD))
        J2_ = -1j*zeros((DD, DD))
        Γ2_ = -1j*zeros((DD, DD))
        for i_ in range(len(shapes)):
            for j_ in range(len(shapes)):
                i, j = i_+nulls, j_+nulls

                J1_ij = F1(i,j)
                J2_ij = F2(i, j)
                Γ2_ij =  Γ2(i, j)

                J1_[ind(i_), ind(j_)] = J1_ij.reshape(prod(J1_ij.shape[:2]), -1)
                J2_[ind(i_), ind(j_)] = J2_ij.reshape(prod(J2_ij.shape[:2]), -1)
                Γ2_[ind(i_), ind(j_)] = Γ2_ij.reshape(prod(Γ2_ij.shape[:2]), -1)

        if not real_matrix:
            return J1_, J2_, Γ2_
        if not as_matrix:
            def gauge(G, i, j):
                return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@c(vL(j)), inv(ch(r(i))), inv(ch(r(j)))],
                            [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])
            def gauge_(G, i, j):
                return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@vL(j), inv(ch(r(i))), inv(ch(r(j)))],
                            [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])

            return (lambda i, j: gauge(F1(i, j), i, j)), (lambda i, j: gauge_(F2(i, j)+Γ2(i, j), i, j))

        J2_ = J2_ + Γ2_

        J = kron(Sz, re(J2_)) + kron(eye(2), re(J1_)) + kron(Sx, im(J2_)) + kron(-1j*Sy, im(J1_))
        return J

    def F1(self, i_, j_, H, envs=None, inv_envs=None, prs_vLs=None, fullH=False, testing=False, id=None):
        '''<d_iψ|H|d_jψ>
        '''
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

        if inv_envs is not None:
            icr, icl, cr, cl = inv_envs
        else:
            icr, icl = [inv(ch(r(i))) for i in range(self.L)], [inv(ch(l(i))) for i in range(self.L)]
            cr, cl = [ch(r(i)) for i in range(self.L)], [ch(l(i)) for i in range(self.L)]

        def inv_ch_r(n): return icr[n]
        def inv_ch_l(n): return icl[n]
        def ch_r(n): return cr[n]
        def ch_l(n): return cl[n]

        if not fullH:
            i, j = (j_, i_) if j_<i_ else (i_, j_)
            gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
            gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1
            G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))
            d, Din_1, Di = self[i].shape

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
                            O_ = ncon([ch_l(m-1)@Am, Am_1]+[h]+[c(vL(m)), inv_ch_r(m)@c(Am_1)],
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

    def F2(self, i_, j_, H, envs=None, inv_envs=None, prs_vLs=None, fullH=False, testing=False, id=None):
        '''<d_id_j ψ|H|ψ>
        '''
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
                           [2, 3, 0, 1])

        L, d, A = self.L, self.d, self.data
        l, r = self.get_envs() if envs is None else envs
        prs_vLs = [self.left_null_projector(n, l, get_vL=True) for n in range(self.L)] if prs_vLs is None else prs_vLs
        def pr(n): return prs_vLs[n][0]
        def vL(n): return prs_vLs[n][1]
        if inv_envs is not None:
            icr, icl, cr, cl = inv_envs
        else:
            icr, icl = [inv(ch(r(i))) for i in range(self.L)], [inv(ch(l(i))) for i in range(self.L)]
            cr, cl = [ch(r(i)) for i in range(self.L)], [ch(l(i)) for i in range(self.L)]

        def inv_ch_r(n): return icr[n]
        def inv_ch_l(n): return icl[n]
        def ch_r(n): return cr[n]
        def ch_l(n): return cl[n]

        i, j = (j_, i_) if j_<i_ else (i_, j_)

        if i==j:
            if testing:
                G = 1j*zeros((*A[i].shape, *A[j].shape))

            gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
            gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1

            G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))
        elif not fullH:
            H = [h.reshape(*[self.d]*4) for h in H]
            # new stuff
            gDi, gDi_1 = vL(i).shape[-1], A[i+1].shape[1] if i != self.L-1 else 1
            gDj, gDj_1 = vL(j).shape[-1], A[j+1].shape[1] if j != self.L-1 else 1
            G_ = 1j*zeros((gDi, gDi_1, gDj, gDj_1))

            d, Din_1, Di = self[i].shape
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

        if testing:
            def gauge(G, i, j):
                return ncon([G, inv(ch(l(i-1)))@vL(i), inv(ch(l(j-1)))@vL(j), inv(ch(r(i))), inv(ch(r(j)))],
                            [[1, 3, 2, 4], [-1, -2, 1], [-4, -5, 2], [-3, 3], [-6, 4]])
            assert allclose(gauge(G_, i, j), G)
            G = tra(G, [3, 4, 5, 0, 1, 2]) if j_<i_ else G
            return G

        G_ = tra(G_, [2, 3, 0, 1]) if j_<i_ else G_
        self.F2_tot_ij_mem[str(i_)+str(j_)] = G_
        return G_

    def christoffel(self, i, j, k, envs=None, prs_vLs=None, id=None, testing=True, closed=(None, None, None)):
        """christoffel: return the christoffel symbol in basis c(A_i), c(A_j), A_k.
           Close indices i, j, k, with elements of closed tuple: i.e. (B_i, B_j, B_k).
           Will leave any indices marked none open :-<d_id_jψ|d_kψ>"""
        id = id if id is not None else uuid.uuid4().hex
        if self.id__ != id:
            self.id__ = id
            # initialize the memories
            # we only don't try the cache on the first call from jac
            self.christ_ij_mem = {}
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

        contracted = False
        if closed[-1] is not None and closed[0] is None and closed[1] is None:
            contracted = True

        def Γ(i_, j_, k):
            """Γ: Christoffel symbol: does take advantage of gauge"""
            #j always greater than i (Γ symmetric in i, j)
            i, j = (j_, i_) if j_<i_ else (i_, j_)
            _, Din_1, Di = self[i].shape

            if j==i or i!=k or (d*Din_1==Di):
                if contracted:
                    G = 1j*zeros((*A[i].shape, *A[j].shape))
                else:
                    G = 1j*zeros((*A[i].shape, *A[j].shape, *A[k].shape))
            else:
                if contracted:
                    if str(i)+str(j) not in self.christ_ij_mem:
                        if hasattr(self, 'dA_cache') and str(j) in self.dA_cache:
                            Rs = self.dA_cache[str(j)]
                        else:
                            R = ncon([pr(j), A[j]], [[-3, -4, 1, -2], [1, -1, -5]])
                            Rs = self.left_transfer(R, i, j)
                            self.christ_ij_mem[str(i)+str(j)] = Rs
                    else:
                        Rs = self.christ_ij_mem[str(i)+str(j)]

                    G = ncon([pr(i), closed[-1]@inv(r(i)), Rs(i+1), inv(r(i))], [[-1, -2, 1, 2], [1, 2, 3], [3, 4, -4, -5, -6], [4, -3]],
                             [1, 2, 3, 4])
                else:
                    R = ncon([pr(j), A[j]], [[-3, -4, 1, -2], [1, -1, -5]])
                    Rs = self.left_transfer(R, i, j)
                    G = ncon([pr(i), Rs(i+1), inv(r(i)), inv(r(i))], [[-1, -2, -7, -8], [1, 2, -4, -5, -6], [1, -9], [2, -3]])

            if not contracted:
                G = -tra(G, [3, 4, 5, 0, 1, 2, 6, 7, 8]) if j_<i_ else -G
            else:
                G = -tra(G, [3, 4, 5, 0, 1, 2]) if j_<i_ else -G

            return G

        # contracted = True if we've already done the contractions in Γ
        if not contracted and any([c is not None for c in closed]):
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
        if contracted:
            return ungauge(Γ_c, i, j, (True, True))
        else:
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

    def tangent_space_dims(self, l=None, get_vLs=False, vLs=None):
        l, _ = self.get_envs() if l is None else (l, None)
        vLs = [self.left_null_projector(n, l=l, get_vL=True)[1] for n in range(self.L)] if vLs is None else vLs
        shapes = [(vL.shape[-1], self.data[n+1].shape[1] if n+1<self.L else 1)
                  for vL, n in zip(vLs, range(self.L))]
        if get_vLs:
            return vLs, shapes
        else:
            return shapes

    def tangent_space_basis(self, type='eye', H=None):
        """ return a tangent space basis
        """
        if type=='eye' or type=='rand':
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
        elif type=='F2':
            if H is None:
                raise Exception
            J1, J2, F = self.jac(H, real_matrix=False)
            J2 = J2+F
            J1 = kron(eye(2), re(J1))+kron(-1j*Sy, im(J1))
            J2 = kron(Sz, re(J2)) + kron(Sx, im(J2))
            l, V = eig(J2)
            idx = l.argsort()
            return V[:, idx]

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
