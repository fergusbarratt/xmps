import warnings
import unittest
import numpy as np
from numpy import dot, diag, empty_like, allclose, squeeze, isclose, stack, sqrt
from numpy import identity, array, zeros, tensordot, reshape, swapaxes, ravel
from numpy import split as chop, concatenate, expand_dims, empty, ones, pad
from numpy import insert, transpose, ones_like
from numpy.random import randn
from numpy.linalg import svd, norm, inv
from scipy.sparse.linalg import eigsh as lanczos
from functools import reduce
from itertools import product
from copy import copy
from qutip import sigmam, sigmap, sigmax, sigmay, sigmaz, tensor
from scipy.linalg import expm
import matplotlib.pyplot as plt
O = zeros((2, 2))
I = identity(2)
sm = sigmam().full()
sp = sigmap().full()
sx = sigmax().full()
sy = sigmay().full()
sz = sigmaz().full()


def S_n_body(op, i, n, tensorise=True):
    '''n_body versions of local operators replace tensor with
        np.array for array of objects'''
    if not i < n:
        raise Exception("i must be less than n")
    l = [op**m for m in map(lambda j: int(not i-j), range(n))]
    if tensorise:
        return tensor(l)
    else:
        return np.array([d.full() for d in l])


def Sp(i, n, tensorise=True):
    return S_n_body(sigmap(), i, n, tensorise)


def Sm(i, n, tensorise=True):
    return S_n_body(sigmam(), i, n, tensorise)


def Sx(i, n, tensorise=True):
    return S_n_body(sigmax(), i, n, tensorise)


def Sy(i, n, tensorise=True):
    return S_n_body(sigmay(), i, n, tensorise)


def Sz(i, n, tensorise=True):
    return S_n_body(sigmaz(), i, n, tensorise)


def structure(l):
    return list(map(lambda x: x[0].shape, l))


def is_right_canonical(mats):
    Is = [sum(map(lambda x_y: dot(x_y[0], x_y[1].conj().T),
                  zip(*[mats[k]]*2)))
          for k in range(len(mats))]
    close = list(map(lambda x: allclose(identity(max(x.shape)), x), Is))
    return array(close).all()


def is_left_canonical(mats):
    Is = [sum(map(lambda x_y: dot(x_y[0].conj().T, x_y[1]),
                  zip(*[mats[k]]*2)))
          for k in range(len(mats))]
    close = list(map(lambda x: allclose(identity(x.shape[0]), x), Is))
    return(array(close).all())


class mps(object):
    '''lists of numpy arrays (1d) of numpy arrays (2d).  generates and
    represents the canonical decompositions of an arbitrary quantum state,
    canonicalise arbitrary state'''

    def __init__(self, state=None, gen=True, hand='right',  # noqa
                 compress=False, D=None, split_point=None, track_singulars=True,
                 mats=None, d=None, use_test_state=False, test_state_d=2,
                 test_state_L=12, test_state_D=10, decompose_test_state=False):
        '''provide either a state (with gen=True, and a hand) or mats, with d
        dimension of local state space. decompose_test_state=True generates
        full d**L tensor and decomposes it according to hand. False generates
        random mps with D=test_state_D'''

        # keep the singular matrices
        self.track_singulars = track_singulars
        self.singulars = []

        if use_test_state is True:
            state = randn(*[test_state_d]*test_state_L) + \
                    1j*randn(*[test_state_d]*test_state_L)
            state = state/norm(state)

        # Generate matrices
        if mats is None and gen and state is not None and not use_test_state or\
           mats is None and gen and use_test_state and decompose_test_state:
            self.state = state

            # Length of chain is number of tensor dimensions
            self.L = self.state.ndim

            # Assume tensor is in product of spaces with same hs dimension
            self.d = self.state.shape[0]

            if D is None and compress is True:
                raise Exception("can't compress without bond dimension")

            if hand is 'left':
                self.hand = hand
                self._left_generate()
            elif hand is 'right':
                self.hand = hand
                self._right_generate()
            elif hand is 'mixed':
                if split_point is None:
                    raise Exception('No split point: can\'t generate mixed')
                self.hand = hand
                self._mixed_generate(split_point)
            else:
                raise NotImplementedError('no other types')
        elif mats is None and gen and use_test_state and \
                not decompose_test_state:
            self.state = None
            self._random_generate(test_state_L, test_state_d, test_state_D)
        else:
            if mats is None:
                raise NotImplementedError('need either to generate or to\
                                          be given matrices')
            elif d is None:
                raise NotImplementedError('provide dimension of local state\
                                          space')
            elif mats is not None and d is not None:
                self.state = None
                self.d = d
                self.L = len(mats)
                self.mps = mats

    def __getitem__(self, index):
        return self.mps[index]

    def __add__(self, y):
        return mps(mats=self.mps + y.mps, d=self.d)

    def _left_generate(self, norm=True):
        Psi = self.state

        def split(Psi, n):
            Psi = Psi.reshape(-1, self.d**(self.L-(n+1)), order='F')
            (U, S, V) = svd(Psi, full_matrices=False)
            if self.track_singulars:
                self.V = V
                self.S = diag(S)
                self.singulars.append(self.S)
            if n == self.L - 1 and norm:
                S = ones_like(S)
            return (chop(U, self.d), dot(diag(S), V))

        As = [None]*self.L
        for n in range(self.L):
            (As[n], Psi) = split(Psi, n)
        assert len(As) == self.L

        self.mps = list(map(lambda x: array(x), As))

        return As

    def _right_generate(self, norm=True):
        Psi = self.state

        def split(Psi, n):
            Psi = Psi.reshape(self.d**(n-1), -1, order='C')
            (U, S, V) = svd(Psi, full_matrices=False)
            if self.track_singulars:
                self.U = U
                self.S = diag(S)
                self.singulars.append(copy(self.S))
            if n == 1 and norm:
                S = ones_like(S)
            return (dot(U, diag(S)), chop(V, self.d, axis=1))

        Bs = [None]*(self.L+1)
        for n in reversed(range(1, self.L+1)):
            # Exclusive of up_to, generate up_to matrices
            (Psi, Bs[n]) = split(Psi, n)
        Bs = Bs[1:]

        self.mps = Bs

        self.mps = list(map(array, Bs))

        return Bs

    def _mixed_generate(self, k):
        assert k < self.L
        assert k >= 0
        self.k = k
        right = self._right_generate()
        self.mps = self.left_canonicalise(mats=right, up_to=k).mps

        return self.mps

    def _recombine(self):
        mps = copy(self.mps)

        if self.state is None:
            state = empty([self.d]*self.L)
        else:
            state = empty_like(self.state)

        for ij in product(*[range(self.d)]*self.L):
            warnings.simplefilter('ignore')
            state[ij] = reduce(dot, map(lambda k_x: k_x[1][ij[k_x[0]]],
                                        enumerate(mps)))[0][0]
        return state

    def _random_generate(self, L, d, D):
        '''generate a random matrix product state with structure
        self._create_structure(L, d, D)'''
        self.L = L
        self.d = d
        self.D = D
        mps = [randn(*((d,) + shape))
               for shape in self._create_structure(L, d, D)]
        self.mps = mps
        return self

    def _create_structure(self, L, d, D):
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

    def append(self, new):
        self.mps.append(new)
        return self

    def delete(self, index):
        del self.mps[index]
        return self

    def to_gamma_lambda(self):
        assert self.track_singulars
        self.left_canonicalise(inplace=True)
        assert self.is_left_canonical()
        Gammas = [self.mps[0]]
        for A, Lambda in zip(self.mps[1:], self.singulars):
            Gammas.append(swapaxes(tensordot(inv(Lambda), A, (-1, 1))[0],
                                   0, 1))
        return list(zip(Gammas, self.singulars))

    def from_gamma_lambda(self, GL):
        Bs = [tensordot(Gamma, Lambda[0], (-1, 0)) for Gamma, Lambda in GL]
        return Bs

    def one_site_bipartition(self, k):
        assert k < self.L
        assert k >= 0
        return (mps(mats=self.mps[:k], d=self.d), self.mps[k],
                mps(mats=self.mps[k+1:], d=self.d))

    def two_site_bipartition(self, k):
        '''k is first site'''
        assert k+1 < self.L
        assert k >= 0
        return (mps(mats=self.mps[:k], d=self.d),
                swapaxes(tensordot(self.mps[k], self.mps[k+1], (-1, 1)), 0, 1),
                mps(mats=self.mps[k+2:], d=self.d))

    def left_canonicalise(self, mats=None, up_to=None, inplace=False, norm=True,
                          compress=False, D=None, mode='svd'):
        '''does thin SVD: can change structure of mps'''
        if mats is None:
            new_mps = copy(self.mps)
        else:
            new_mps = copy(mats)
        if up_to is None:
            k = len(self.mps) + 1
        else:
            k = up_to
        if self.track_singulars:
            self.singulars = []

        def split(M, m):
            (A, S, V) = svd(M, full_matrices=False)
            if norm and m == len(self.mps)-1:
                S = ones_like(S)
            if self.track_singulars:
                self.singulars.append(array([diag(S)]))
            return (array(chop(A, self.d, axis=0)), diag(S), V)

        def truncate(A, S, V, D):
            if D is not None:
                A_trunc = min(A.shape[2], D)
                S_trunc = min(S.shape[0], D)
                V_trunc = min(V.shape[0], D)
                A = A[:, :, :A_trunc]
                S = S[:S_trunc, :S_trunc]
                V = V[:V_trunc, :]
            return (A, dot(S, V))

        for m in range(len(new_mps))[:k]:
            A, S, V = split(concatenate(new_mps[m], axis=0), m)
            A, SV = truncate(A, S, V, D)
            new_mps[m] = A
            if m+1 < len(new_mps):
                new_mps[m+1] = array(list(map(lambda x: SV.dot(x),
                                              new_mps[m+1])))

        new_mps = list(map(array, new_mps))
        if inplace:
            self.mps = new_mps
            return self
        else:
            return mps(mats=new_mps, d=self.d)

    def right_canonicalise(self, mats=None, up_to=None, inplace=False,
                           norm=True, D=None):
        '''right canonicalises internal mps (or external) up_to given site,
        compresses via svd to bond dimension D unless D is none.'''
        if mats is None:
            new_mps = copy(self.mps)
        else:
            new_mps = copy(mats)
        if up_to is None:
            k = 0
        else:
            k = up_to
        if self.track_singulars:
            self.singulars = []

        def split(M):
            (U, S, B) = svd(M, full_matrices=False)
            if norm and m == 0:
                S = ones_like(S)
            if self.track_singulars:
                self.singulars.append(diag(S))
            return (U, diag(S), array(chop(B, self.d, axis=1)))

        def truncate(U, S, B, D):
            if D is not None:
                U_trunc = min(U.shape[1], D)
                S_trunc = min(S.shape[0], D)
                B_trunc = min(B.shape[1], D)
                U = U[:, :U_trunc]
                S = S[:S_trunc, :S_trunc]
                B = B[:, :B_trunc, :]
            return (dot(U, S), B)

        for m in range(len(new_mps))[k:][::-1]:
            U, S, B = split(concatenate(new_mps[m], axis=1))
            US, new_mps[m] = truncate(U, S, B, D)
            if m-1 >= 0:
                new_mps[m-1] = array(list(map(lambda x: x.dot(US),
                                              new_mps[m-1])))

        new_mps = list(map(array, new_mps))
        if inplace:
            self.mps = new_mps
            return self
        else:
            return mps(mats=new_mps, d=self.d)

    def compress(self, D=None, mode='svd'):  # noqa
        '''compresses to bond dimension D during a right canonicalisation'''

        new_mps = copy(self.mps)

        self.D = max([x for y in self.structure() for x in y])

        self.truncation_error = 0

        if D is None or D > self.D:
            return self

        new_mps = self.left_canonicalise(new_mps).mps

        def split(M, m):
            (U, S, B) = svd(M, full_matrices=False)
            return (U, diag(S), array(chop(B, self.d, axis=1)))

        def truncate(U, S, B, D):
            if D is not None:
                e = 1 - norm(S[:D, :D])
                U = U[:, :D]
                S = S[:D, :D]
                B = B[:, :D, :]
            return (dot(U, S), B, e)

        for m in range(len(new_mps))[::-1]:
            U, S, B = split(concatenate(new_mps[m], axis=1), m)
            US, new_mps[m], e = truncate(U, S, B, D)
            if m-1 >= 0:
                new_mps[m-1] = array(list(map(lambda x: x.dot(US),
                                                new_mps[m-1])))
            self.truncation_error += e

        new_mps = list(map(array, new_mps))

        self.mps = new_mps
        return self

    def is_right_canonical(self):
        Is = [sum(map(lambda x_y: dot(x_y[0], x_y[1].conj().T),
                      zip(*[self.mps[k]]*2)))
              for k in range(self.L)]
        close = list(map(lambda x: allclose(identity(max(x.shape)), x), Is))
        return array(close).all()

    def is_left_canonical(self):
        Is = [sum(map(lambda x_y: dot(x_y[0].conj().T, x_y[1]),
                      zip(*[self.mps[k]]*2)))
              for k in range(self.L)]
        close = list(map(lambda x: allclose(identity(x.shape[0]), x), Is))
        return(array(close).all())

    def structure(self):
        return list(map(lambda x: x[0].shape, copy(self.mps)))

    def overlap(self, state, pad_state=True):
        assert len(self.mps) == len(state.mps)
        padded_self = copy(self.mps)
        padded_state = copy(state.mps)

        def conj(list_of_matrices):
            new_list = []
            for matrix in list_of_matrices:
                new_list.append(matrix.conj().T)
            return array(new_list)

        if not self.structure() == state.structure():
            for index, (shapes, (self_matrices, state_matrices)) in enumerate(
                                                   zip(zip(self.structure(),
                                                           state.structure()),
                                                       zip(self.mps,
                                                           state.mps))):
                pads = insert(array(shapes[1])-array(shapes[0]), 0, 0)
                if (pads < 0).any():
                    assert not (pads > 0).any()
                    pads = list(zip([0, 0, 0], -pads))
                    padded_state[index] = pad(state_matrices, pads,
                                              mode='constant')
                else:
                    pads = list(zip([0, 0, 0], pads))
                    padded_self[index] = pad(self_matrices, pads,
                                             mode='constant')

        assert structure(padded_self) == structure(padded_state)
        F = tensordot(conj(padded_self[0]), padded_state[0], ([0, 1], [0, 2]))
        for L_d, R in zip(padded_self, padded_state):
            L = conj(L_d)
            F = tensordot(tensordot(L, F, (-1, 0)), R, ([0, -1], [0, 1]))
        return F[0][0]

    def tDMRG_evolve(self, H, dt, steps=5, D=None):
        '''Do tDMRG time evolution. Just does one time step
        Not sure the extent to which this works'''
        self.initial_state = copy(self)
        left = self.left_canonicalise()

        self.mps = self.right_canonicalise(mats=left.mps, up_to=2).mps
        (As, Psi, Bs) = self.two_site_bipartition(1)

        odds = range(1, len(self.mps)-3)[::2]

        def shift(As, Psi, Bs, dt):
            Phi = tensordot(
                expm(-1j * H[site] * dt),
                reshape(
                    Psi,
                    (Psi.shape[0] * Psi.shape[1],
                     Psi.shape[2],
                     Psi.shape[3])),
                (-1, 0))
            Phi = stack(chop(Phi, self.d, axis=0), axis=0)

            def split(M):
                (A, S, V) = svd(M, full_matrices=False)
                return (array(chop(A, self.d, axis=0)), diag(S), V)

            def truncate(A, S, V, D):
                if D is not None:
                    A = A[:, :, :D]
                    S = S[:D, :D]
                    V = V[:D, :]
                return (A, dot(S, V))

            to_split = reshape(transpose(Phi, [2, 0, 1, 3]),
                               (Phi.shape[0]*Phi.shape[2], -1))

            A1, S, V = split(to_split)
            A1, SV = truncate(A1, S, V, D)

            SV = swapaxes(reshape(SV, (SV.shape[0], -1, Bs[0].shape[1])), 0, 1)

            SVB = transpose(tensordot(SV, Bs[0], (-1, 1)), [1, 0, 3, 2])

            Phi = reshape(SVB, (SVB.shape[0]*SVB.shape[1], -1))

            A2, S, V = split(Phi)
            A2, SV = truncate(A2, S, V, D)

            SV = swapaxes(reshape(SV, (SV.shape[0], -1, Bs[1].shape[1])), 0, 1)

            Phi = swapaxes(tensordot(SV, Bs[1], (-1, 1)), 1, 2)
            return (As.append(A1).append(A2), Phi, Bs.delete(0).delete(0))

        dt = t/steps
        for site in odds:
            As, Psi, Bs = shift(As, Psi, Bs, dt)

        return (As, Psi, Bs)

    def tMPS_evolve(self, H, dt, steps=5, D=200):
        '''receive a hamiltonian of form [h_i for i in range(n)] (H=sum([...]))
        (where h_i is nearest neighbour (i-> i+1) and evolve MPS under
        that hamiltonian. truncate to D after each time step'''

        for i, h in enumerate(H):
            u = expm(-1j*h*dt)
            u = stack(chop(u, self.d, axis=0), axis=0)
            u = stack(chop(u, self.d, axis=2), axis=-1)
            u = transpose(u, [0, 3, 1, 2])
            p = reshape(u, (self.d**2, self.d**2))
            U, S, V = svd(p, full_matrices=False)
            U1 = stack(chop(dot(U, sqrt(diag(S))), 2, axis=0), axis=0)
            U2 = stack(chop(dot(sqrt(diag(S)), V), 2, axis=-1), axis=-1)
            U1 = expand_dims(U1, 2)
            U2 = expand_dims(U2, 1)
            U2 = transpose(U2, [3, 2, 0, 1])
            H[i] = (U1, U2)

        id = expand_dims(expand_dims(identity(self.d), -1), -1)

        if self.L % 2 == 0:
            odds = mpo([x for y in H[::2] for x in y])
            evens = mpo([id]+[x for y in H[1::2] for x in y]+[id])
        else:
            odds = mpo([x for y in H[::2] for x in y]+[id])
            evens = mpo([id]+[x for y in H[1::2] for x in y])

        new_state = self
        self.initial_state = copy(self)
        overlaps = []

        error = 0
        for i in range(steps):
            overlaps.append(self.initial_state.overlap(new_state))
            new_state = evens.apply(odds.apply(new_state))
            new_state = new_state.compress(D)
            error += new_state.truncation_error

        return new_state

    def TEBD_evolve(self, H, dt, steps, D=None, verbose=0):
        GL = self.to_gamma_lambda()

        def new_Psi(l):  # si_{l+1}, si_{l+2}
            '''works by magic (dot does [1, -2] which is the first mps
            index of G)'''
            l = l-1

            Psi = dot(dot(dot(dot(squeeze(GL[l][1]), GL[l+1][0]),
                              squeeze(GL[l+1][1])),
                          GL[l+2][0]),
                      squeeze(GL[l+2][1]))
            return transpose(Psi, [1, 2, 0, 3])

        for _ in range(steps):
            for site in range(1, self.L-1):
                Psi = new_Psi(site)
                h = H[site]

                Phi = tensordot(
                    expm(-1j * h * dt),
                    reshape(Psi, (Psi.shape[0] * Psi.shape[1],
                                  Psi.shape[2], Psi.shape[3])), (-1, 0))
                Phi = stack(chop(Phi, self.d, axis=0), axis=0)

                def split(M):
                    (A, S, V) = svd(M, full_matrices=False)
                    return (array(chop(A, self.d, 0)), diag(S), V)

                def truncate(A, S, V, D):
                    if D is not None:
                        A = A[:, :, :D]
                        S = S[:D, :D]
                        V = V[:D, :]
                    return (A, S, V)

                to_split = reshape(transpose(Phi, [2, 0, 1, 3]),
                                   (Phi.shape[0]*Phi.shape[2], -1))

                U, L_l_1, V = truncate(*split(to_split), D)  # noqa
                U = swapaxes(reshape(U, (-1, self.d, U.shape[-1])), 0, 1)
                V = swapaxes(reshape(V, (V.shape[0], self.d, -1)), 0, 1)

                L_l = GL[site-1][1]
                if site is not 1:
                    L_l = expand_dims(L_l, axis=0)

                L_l_2 = GL[site+1][1]
                G_l_1 = squeeze(dot(inv(L_l), U))
                G_l_2 = squeeze(dot(V, inv(L_l_2)))

                if site is self.L-2:
                    G_l_2 = expand_dims(G_l_2, axis=-1)

                GL[site] = (swapaxes(G_l_1, 0, 1), expand_dims(L_l_1, 0))
                GL[site+1] = (G_l_2, L_l_2)

        return mps(mats=self.from_gamma_lambda(GL),
                   d=self.d).left_canonicalise()


class mpo(object):
    '''list of arrays of numpy arrays. (like MPS but two indices per site)'''

    def __init__(self, mats=None, s182=False, length=None, h=1, J=1, Jz=1):
        if s182 and length is not None:
            self.L = length
            self.mpo = self.s182(h, J, Jz)
        elif not s182:
            self.mpo = mats
            self.L = len(mats)
        elif s182 and length is None:
            raise Exception("Need a length to generate s182")
        self.h, self.J, self.Jz = h, J, Jz

    def s182(self, h, J, Jz):
        first = array([[-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
        bulk = array([[I, O, O, O, O],
                      [sp, O, O, O, O],
                      [sm, O, O, O, O],
                      [sz, O, O, O, O],
                      [-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
        last = array([[I], [sp], [sm], [sz], [-h*sz]])
        mpo = [first]+[bulk]*(self.L-2)+[last]
        for i, mats in enumerate(mpo):
            mpo[i] = np.swapaxes(np.swapaxes(mats, 0, 2), 1, 3)
        return mpo

    def id_sum(self):
        return [identity(h.shape[0]) for h in self.s182_sum()]

    def s182_sum(self):
        J, Jz, h = self.J, self.Jz, self.h
        H = [J/2*(Sp(0, 2)*Sm(1, 2) +
                  Sm(0, 2)*Sp(1, 2)) +
             Jz*Sz(0, 2)*Sz(1, 2) -
             h*Sz(0, 2)
             for _ in range(self.L-1)]
        H[-1] = H[-1] - h*Sz(1, 2)
        return [h_i.full() for h_i in H]

    def mixed_apply(self, mps):
        '''apply a hamiltonian mpo operator to a mixed canonical mps'''
        assert mps.hand is 'mixed'
        assert mps.L == self.L
        l = mps.k  # the position of the non canonical matrix in the mps

        # generate HPsi via correct bracketing
        # Iterate L and R
        L = self.gen_L(l-1, mps)
        R = self.gen_R(l, mps)

        # mpo and mps at split site
        Psi = mps[l]
        W = self.mpo[l]

        # perform the contractions that schollwock eq.199 suggests.
        RPsi = tensordot(R, Psi, (2, 1))
        WRPsi = tensordot(W, RPsi, ([3, 1], [1, 2]))
        LWRPsi = tensordot(L, WRPsi, ([1, 2], [1, 2]))
        # still has indices (a_{l-1}, sigma_l, a_l) for left block,
        # site and right block bases resp.

        HPsi = LWRPsi

        return HPsi

    def apply(self, state):
        '''Not sure about reshaping here'''
        new_mps = copy(state.mps)
        for i, (o, s) in enumerate(zip(self.mpo, state.mps)):
            new_mps[i] = reshape(swapaxes(tensordot(o, s, (1, 0)), 2, 3),
                                 (state.d, -1, o.shape[-1]*s.shape[-1]))

        new_mps = list(map(array, new_mps))

        return mps(mats=new_mps, d=state.d)

    def gen_L(self, up_to, mps, scan=False):
        """gen_L

        :param up_to:
        :param mps:
        :param scan:
        """
        W = self.mpo[0]
        A = mps[0]
        A_d = array(list(map(lambda x: x.conj().T, A)))
        WA = np.squeeze(tensordot(W, A, (1, 0)))
        F1 = np.squeeze(tensordot(A_d, WA, (0, 0)))

        # Generic update
        def update_l(F, i):
            '''update F_i-1 to F_i'''
            W = self.mpo[i]
            A = mps[i]
            A_d = array(list(map(lambda x: x.conj().T, A)))
            FA = tensordot(F, A, (2, 1))
            WFA = tensordot(W, FA, ([1, 2], [2, 1]))
            AdWFA = tensordot(A_d, WFA, ([0, 2], [0, 2]))
            return AdWFA

        # Apply update from site 1 up to and including site n
        def update_n_l(F1, n, scan):
            if not scan:
                Fn = F1
                for i in range(1, n+1):
                    Fn = update_l(Fn, i)
                return Fn
            else:
                Fns = [copy(F1)]
                for i in range(1, n+1):
                    Fns.append(update_l(Fns[-1], i))
                return Fns

        return update_n_l(F1, up_to, scan=scan)

    def gen_R(self, up_to, mps, scan=False):
        '''Generate matrix R iteratively'''
        # Initial
        W = self.mpo[-1]
        B = mps[-1]
        B_d = array(list(map(lambda x: x.conj().T, B)))
        WB = np.squeeze(tensordot(W, B, (0, 0)))
        FL = np.squeeze(tensordot(B_d, WB, (0, 0)))

        # Generic update
        def update_r(F, i_min_1):
            '''update F_i to F_i-1 (also take 1 because of indexes)'''
            W = self.mpo[i_min_1-1]
            B = mps[i_min_1-1]
            B_d = array(list(map(lambda x: x.conj().T, B)))
            FBd = tensordot(F, B_d, (2, 1))
            WFBd = tensordot(W, FBd, ([0, 3], [2, 1]))
            BWFBd = tensordot(B, WFBd, ([0, 2], [0, 2]))
            return BWFBd

        # Update from site L up to and including site L-n
        def update_n_r(FL, n, scan):
            '''(self.L-(self.L-l))=l'''
            if not scan:
                Fn = copy(FL)
                for i in range(1, n+1):
                    Fn = update_r(Fn, self.L-i)
                return Fn
            else:
                Fns = [copy(FL)]
                for i in range(1, n+1):
                    Fns.append(update_r(Fns[-1], self.L-i))
                return Fns[::-1]

        return update_n_r(FL, self.L - (up_to+1), scan)

    def ground_state(self, iters=5, D=None, initial_state=None,  # noqa
                     return_state=False, verbose=1):
        '''find the ground state of mpo hamiltonian. algorithm is p148-9
        schollwock'''
        if initial_state is None:
            if verbose:
                print('\ngenerating state')
            initial_state = mps(gen=True,
                                use_test_state=True,
                                test_state_L=self.L,
                                test_state_d=2,
                                test_state_D=D).right_canonicalise()
            if verbose:
                print('generated state')

        assert initial_state.L == self.L
        gs = []
        state = initial_state

        # Generic update(replicated)
        def update_l(F, i, mps):
            '''update F_i-1 to F_i'''
            W = self.mpo[i]
            A = mps[i]
            A_d = array(list(map(lambda x: x.conj().T, A)))
            FA = tensordot(F, A, (2, 1))
            WFA = tensordot(W, FA, ([1, 2], [2, 1]))
            AdWFA = tensordot(A_d, WFA, ([0, 2], [0, 2]))
            return AdWFA

        def update_r(F, i_min_1, mps):
            '''update F_i to F_i-1 (also take 1 because of indexes)'''
            W = self.mpo[i_min_1-1]
            B = mps[i_min_1-1]
            B_d = array(list(map(lambda x: x.conj().T, B)))
            FBd = tensordot(F, B_d, (2, 1))
            WFBd = tensordot(W, FBd, ([0, 3], [2, 1]))
            BWFBd = tensordot(B, WFBd, ([0, 2], [0, 2]))
            return BWFBd

        # Initialise
        state = initial_state

        def right_sweep():
            # generate R matrices
            Rs = self.gen_R(up_to=1, mps=state, scan=True)

            # Sweep right, restoring normalisation at each step
            # 1st site

            M = initial_state[0]

            L = ones((1, 1, 1))
            W = self.mpo[0]
            R = Rs[0]

            WR = tensordot(W, R, (3, 1))  # (si_l, si_l', a_l', a_l)
            LWR = tensordot(L, WR, (1, 2))
            # (a_{l-1}', a_{l-1}, si_l, si_l', a_l,      a_l')
            # (si_l,    a_{l-1}, a_l, si_l', a_{l-1}', a_l')
            LWR = transpose(LWR, [2, 1, 4, 3, 0, 5])

            H = reshape(LWR, (LWR.shape[0]*LWR.shape[1]*LWR.shape[2], -1))
            v = ravel(M)
            lambda0, v = lanczos(H, k=1, which='SA', v0=v)
            M_n = v.reshape(M.shape)
            (U, S, V) = svd(M_n.reshape(
                            (M_n.shape[0] * M_n.shape[1], M_n.shape[2])),
                            full_matrices=False)
            SV = dot(diag(S), V)
            A = array(chop(U, state.d, axis=0))

            state.mps[0] = A
            state.mps[1] = array(list(map(lambda x: dot(SV, x), state[1])))
            gs.append(lambda0)

            # replicate gen_L code here (reorg this)
            # Generate matrix L iteratively
            # Initial (always start from site 1)
            W = self.mpo[0]
            A = state[0]
            A_d = array(list(map(lambda x: x.conj().T, A)))
            WA = np.squeeze(tensordot(W, A, (1, 0)))
            F1 = np.squeeze(tensordot(A_d, WA, (0, 0)))

            Ls = [F1]

            for site, M in enumerate(state.mps[1:-1]):
                site = site+1

                L = Ls[-1]
                W = self.mpo[site]
                R = Rs[site]

                WR = tensordot(W, R, (3, 1))  # (si_l, si_l', a_l', a_l)
                LWR = tensordot(L, WR, (1, 2))
                # (a_{l-1}', a_{l-1}, si_l, si_l', a_l,      a_l')
                # (si_l,    a_{l-1}, a_l, si_l', a_{l-1}', a_l')
                LWR = transpose(LWR, [2, 1, 4, 3, 0, 5])

                H = reshape(LWR, (LWR.shape[0]*LWR.shape[1]*LWR.shape[2], -1))
                v = ravel(M)
                lambda0, v = lanczos(H, k=1, which='SA', v0=v)
                M_n = v.reshape(M.shape)
                (U, S, V) = svd(M_n.reshape(
                        (M_n.shape[0] * M_n.shape[1],
                         M_n.shape[2])),
                    full_matrices=False)
                SV = dot(diag(S), V)
                A = array(chop(U, state.d, axis=0))

                state.mps[site] = A
                state.mps[site+1] = array(list(map(lambda x: dot(SV, x),
                                                   state[site+1])))

                gs.append(lambda0)
                Ls.append(update_l(Ls[-1], site, state.mps))

        def left_sweep():
            # generate all Ls
            Ls = self.gen_L(up_to=self.L-2, mps=state, scan=True)

            # Sweep left, restoring normalisation at each step
            # Lth site
            W = self.mpo[-1]
            M = initial_state[-1]
            L = Ls[-1]

            LW = squeeze(tensordot(L, W, (1, 2)))  # (si_l, si_l', a_l', a_l)
            # (a_{l-1}, a_{l-1}', si_l, si_l')
            # (si_l, a_{l-1}', a_{l-1}, si_l') 0->2
            # (si_l, a_{l-1}, a_{l-1}', si_l') 1->2
            # (si_l, a_{l-1}, si_l', a_{l-1}') 2->3

            H = reshape(swapaxes(swapaxes(swapaxes(LW, 0, 2), 1, 2), 2, 3),
                        (LW.shape[0]*LW.shape[1], LW.shape[2]*LW.shape[3]))
            v = ravel(M)

            assert allclose(H, H.conj().T)

            lambda0, v = lanczos(H, k=1, which='SA', v0=v)
            M_n = v.reshape(M.shape)  # (2, 2, 1)

            (U, S, V) = svd(M_n.reshape((M_n.shape[1],
                                         M_n.shape[0]*M_n.shape[2])),
                            full_matrices=False)
            # might need C indexing here
            US = dot(U, diag(S))
            B = array(chop(V, state.d, axis=1))
            state.mps[-1] = B
            state.mps[-2] = array(list(map(lambda x: dot(x, US), state[-2])))
            gs.append(lambda0)

            W = self.mpo[-1]
            B = state[-1]
            B_d = array(list(map(lambda x: x.conj().T, B)))
            WB = np.squeeze(tensordot(W, B, (0, 0)))
            FL = np.squeeze(tensordot(B_d, WB, (0, 0)))

            Rs = [FL]

            for site, M in list(enumerate(state.mps[1:-1]))[::-1]:
                site = site+1  # L-> 2
                if verbose > 1:
                    print(site)

                L = Ls[site-1]
                W = self.mpo[site]
                R = Rs[0]

                WR = tensordot(W, R, (3, 1))  # (si_l, si_l', a_l', a_l)
                LWR = tensordot(L, WR, (1, 2))
                # (a_{l-1}, a_{l-1}', si_l, si_l', a_l,      a_l')
                LWR = swapaxes(LWR, 1, 4)
                # (a_{l-1}, a_l,      si_l, si_l', a_{l-1}', a_l')
                LWR = swapaxes(LWR, 0, 2)
                # (si_l,    a_l,     a_{l-1}, si_l', a_{l-1}', a_l')
                LWR = swapaxes(LWR, 1, 2)
                # (si_l,    a_{l-1}, a_l, si_l', a_{l-1}', a_l')
                LWR = swapaxes(LWR, 2, 5)  # Fixed Bug!!! no idea why
                H = reshape(LWR, (LWR.shape[0]*LWR.shape[1]*LWR.shape[2], -1))
                v = ravel(M)

                assert allclose(H, H.conj().T)

                lambda0, v = lanczos(H, k=1, which='SA', v0=v)
                M_n = swapaxes(v.reshape(M.shape), 0, 1)
                (U, S, V) = svd(M_n.reshape((M_n.shape[0], -1)),
                                full_matrices=False)
                US = dot(U, diag(S))

                B = array(chop(V, state.d, axis=1))

                assert B.shape == state.mps[site].shape
                assert array(list(map(lambda x: dot(x, US), state[
                             site-1]))).shape == state.mps[site-1].shape

                state.mps[site] = B
                state.mps[site-1] = array(
                    list(map(lambda x: dot(x, US), state[site-1])))
                gs.append(lambda0)
                Rs.insert(0, update_r(Rs[0], site+1, state.mps))

        for iteration in range(iters):
            right_sweep()
            left_sweep()
            if verbose:
                print(iteration+1, '/', iters)

        if return_state:
            return gs, state
        else:
            return gs

    def structure(self):
        return list(map(lambda x: x.shape, self.mpo))


class TestMps(unittest.TestCase):

    def setUp(self):
        '''spin chain length L, local dimension d, mixed state is split at s'''
        self.L = np.random.randint(9, 11)
        self.d = 2
        s = np.random.randint(0, self.L-1)
        self.D = np.random.randint(2, 30)

        full = randn(*[self.d]*self.L) + 1j*randn(*[self.d]*self.L)

        self.full = full/norm(full)

        self.l_cmps_svd = mps(self.full, hand='left')
        self.r_cmps_svd = mps(self.full, hand='right')
        self.m_cmps_svd = mps(self.full, hand='mixed', split_point=s)

        self.h = 10*np.random.rand()
        self.J = 10*np.random.rand()
        self.Jz = 10*np.random.rand()
        self.s182 = mpo(s182=True, length=self.L,
                        h=self.h, J=self.J, Jz=self.Jz)

        self.left_cases = [self.l_cmps_svd]
        self.right_cases = [self.r_cmps_svd]
        self.mixed_cases = [self.m_cmps_svd]
        self.recombine_cases = self.right_cases + self.left_cases + \
            self.mixed_cases
        self.cases = self.left_cases + self.right_cases + self.mixed_cases

    def test_recombined_tensor_is_correct(self):
        for case in self.recombine_cases:
            rec = case._recombine()
            self.assertTrue(allclose(rec, self.full) or
                            allclose(rec, -self.full), msg=case.hand)

    def test_left_generated_is_left_canonical(self):
        for case in self.left_cases:
            self.assertTrue(case.is_left_canonical())

    def test_right_generated_is_right_canonical(self):
        for case in self.right_cases:
            self.assertTrue(case.is_right_canonical())

    def test_mixed_generate_is_mixed_canonical(self):
        for case in self.mixed_cases:
            left, center, right = case.one_site_bipartition(case.k)
            self.assertTrue(left.is_left_canonical())
            self.assertTrue(right.is_right_canonical())

    def test_random_generate_has_correct_structure(self):
        for case in self.cases:
            struct1 = case.compress(self.D).structure()
            struct2 = case._random_generate(self.L, self.d, self.D).structure()
            self.assertTrue(struct1 == struct2)

    def test_to_gamma_lambda_works(self):
        for case in self.cases:
            case.to_gamma_lambda()

    def test_right_canonicalised_is_right_canonical(self):
        for case in self.left_cases:
            self.assertTrue(case.is_left_canonical())
            self.assertTrue(not case.is_right_canonical())

            case.right_canonicalise(inplace=True)

            self.assertTrue(not case.is_left_canonical())
            self.assertTrue(case.is_right_canonical())

    def test_left_canonicalised_is_left_canonical(self):
        # right canonical matrices are by definition not left canonical.
        for case in self.right_cases:
            self.assertTrue(case.is_right_canonical())
            self.assertTrue(not case.is_left_canonical())

            case.left_canonicalise(inplace=True)

            self.assertTrue(not case.is_right_canonical())
            self.assertTrue(case.is_left_canonical())

    def test_compressed_has_correct_structure(self):
        s = np.random.randint(3, 20)

        compressed = [case.compress(s) for case in self.cases]
        for case in compressed:
            self.assertTrue(max([x for y in case.structure() for x in y]) <=
                            case.D)

    def test_compressed_preserves_norm(self):
        D = np.random.randint(2, 20)
        for case in self.cases:
            c = copy(case).compress(D)
            self.assertTrue(isclose(c.overlap(c), 1))

    def test_compressed_is_noop_for_D_None(self):
        for case in self.cases:
            c = copy(case).compress(None)
            self.assertTrue(isclose(case.overlap(c), case.overlap(case)))
            self.assertTrue(c.truncation_error == 0)

    def test_create_structure_has_correct_structure(self):
        for case in self.cases:
            struct1 = case._create_structure(self.L, self.d, self.D)
            struct2 = case.compress(self.D).structure()
            if not struct1 == struct2:
                print(struct1)
                print(struct2)
            self.assertTrue(struct1 == struct2)

    def test_structure_is_consistent(self):
        x = 1
        for case in self.cases:
            for dimensions in case.structure():
                self.assertTrue(x == dimensions[0])
                (_, x) = dimensions

    def test_becomes_exact_for_large_bond_dimension(self):
        Ds = range(1, 40)
        test = mps(self.full)
        cases = [copy(test).compress(D) for D in Ds]
        norms = [norm(self.full - case._recombine()) for case in cases]
        plot = False
        if plot:
            plt.plot(array(norms))
            plt.show()

    def test_overlap_gives_correct_norm(self):
        for case in self.cases:
            LR = case.overlap(case)
            self.assertTrue(isclose(LR, 1) or isclose(LR, 2))

    def test_time_evolutions(self):
        dt = -1e-15j
        steps = 4
        D = 100
        for case in self.right_cases:
            state1 = copy(case).tMPS_evolve(self.s182.s182_sum(), dt, steps, D)
            print(state1.overlap(case))


class TestMpo(unittest.TestCase):

    def setUp(self):
        self.L = np.random.randint(4, 16)

        self.d = 2

        self.h = 10*np.random.randn()
        self.J = 10*np.random.randn()
        self.Jz = 10*np.random.randn()
        self.s182 = mpo(s182=True, length=self.L,
                        h=self.h, J=self.J, Jz=self.Jz)

    def test_ground_state_by_exact_diagonalization(self):
        if self.L < 10:
            def spin_half_heisenberg(J, Jz, h, N, tensorise=True):
                NN = sum([J/2*(Sp(i, N, tensorise)*Sm(i+1, N, tensorise) +
                               Sm(i, N, tensorise)*Sp(i+1, N, tensorise)) +
                          Jz*Sz(i, N, tensorise)*Sz(i+1, N, tensorise)
                          for i in range(N-1)])
                S = -sum([h*Sz(i, N) for i in range(N)])
                return NN + S

            An = spin_half_heisenberg(h=self.h, J=self.J, Jz=self.Jz, N=self.L)
            gs_a = An.groundstate()[0]

            iters = 20
            gs_c = self.s182.ground_state(iters=iters, D=10, verbose=0)

            plot = False
            if plot:
                x = range(len(gs_c))
                plt.plot(x, gs_c)
                plt.hlines(gs_a, 0, len(x), colors='r')
                plt.show()

            self.assertTrue(isclose(gs_c[-1], gs_a))
        else:
            print('\nToo large for exact diagonalisation')
            self.assertTrue(True)

    def test_ground_state_converges(self):
        iters = 10
        D = 4
        gs_c = self.s182.ground_state(iters=iters, D=D, verbose=0)
        self.assertTrue(isclose(gs_c[-2], gs_c[-1]))
        plot = True
        if plot:
            x = range(len(gs_c))
            plt.plot(x, gs_c)
            plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=1)

