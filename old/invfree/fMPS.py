import unittest
from numpy.random import rand, randint, randn
from numpy.linalg import svd, inv, norm
from numpy import array, concatenate, diag, dot, allclose, isclose
from numpy import identity, swapaxes, trace, tensordot, sum
from numpy import split as chop, zeros, ones, ones_like, empty
from tests import is_right_canonical, is_right_env_canonical, is_full_rank
from tests import is_left_canonical, is_left_env_canonical, has_trace_1
from tensor import H, truncate_A, truncate_B, diagonalise, rank, mps_pad
from tensor import rdot, ldot, structure
from qmb import sigmaz, sigmax, sigmay
from functools import reduce
from itertools import product


class fMPS(object):
    """finite MPS:
    lists of numpy arrays (1d) of numpy arrays (2d). Finite"""

    def __init__(self, data=None, d=None):
        """__init__

        :param d: local state space dimension
        :param generate_random: if True, generate a random fMPS and ignore data.
        If false use data as data
        :param L: Length
        :param D: Bond dimension
        :param data: if not generate_random, use data as internal fMPS
        """

        if data is not None:
            assert d is not None
            self.L = len(data)
            self.d = d
            self.D = max([x[0].shape for x in data])
            self.data = data

    def __getitem__(self, k):
        """__getitem__

        :param k: item to get
        """
        return self.data[k]

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

    def left_from_state(self, state):
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

        return self

    def right_from_state(self, state):
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
        """create_structure

        :param L: Length
        :param d: local state space dimension
        :param D: bond dimension
        generate the structure of a OBC fMPS
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

    def right_canonicalise(self, D=None, testing=True):
        """right_canonicalise: bring internal fMPS to right canonical form,
        potentially with a truncation

        :param D: bond dimension to truncate to during right sweep
        """
        if D is not None:
            self.D = D

        self.ok = True

        def split(M):
            """split: Do SVD, and reshape B matrix

            :param M: matrices
            """
            (U, S, B) = svd(M, full_matrices=False)
            return (U, diag(S), array(chop(B, self.d, axis=1)))

        # left sweep
        for m in range(len(self.data))[::-1]:
            U, S, B = split(concatenate(self.data[m], axis=1))
            U, S, self.data[m] = truncate_B(U, S, B, D)
            if m-1 >= 0:
                self.data[m-1] = tensordot(self.data[m-1], dot(U, S), (-1, 0))

        # right sweep
        Vs = [None]*self.L
        Ls = [None]*self.L

        V = S = identity(1)
        for m in range(len(self.data)):
            M = trace(dot(dot(dot(dot(H(self.data[m]),
                                  V),
                              S),
                          H(V)),
                      self.data[m]), 0, 0, 2)

            Vs[m], Ls[m] = diagonalise(M)

            self.data[m] = swapaxes(dot(dot(H(V), self.data[m]), Vs[m]),
                                    0, 1)

            V, S = Vs[m], Ls[m]

        Ls = [ones_like(Ls[-1])] + Ls  # Ones on each end

        for m in range(len(self.data)):
            # sort out the rank (tDj is tilde)
            Dj = Ls[m].shape[0]
            tDj = rank(Ls[m])
            P = concatenate([identity(tDj), zeros((tDj, Dj-tDj))], axis=1)
            self.data[m-1] = self.data[m-1] @ H(P)
            self.data[m] = P @ self.data[m]
            Ls[m] = P @ Ls[m] @ H(P)

        self.Ls = Ls  # store all the singular values

        if testing:
            self.update_properties()
            self.ok = self.ok \
                and self.is_right_canonical\
                and self.is_right_env_canonical\
                and self.is_full_rank\
                and self.has_trace_1

        return self

    def left_canonicalise(self, D=None, testing=True):
        """left_canonicalise: bring internal fMPS to left canonical form,
        potentially with a truncation

        :param D: bond dimension to truncate to during right sweep
        """
        if D is not None:
            self.D = D

        self.ok = True

        def split(M):
            """split: Do SVD and reshape A matrix

            :param M: matrix
            """
            (A, S, V) = svd(M, full_matrices=False)
            return (array(chop(A, self.d, axis=0)), diag(S), V)

        for m in range(len(self.data)):
            # sort out canonicalisation
            A, S, V = split(concatenate(self.data[m], axis=0))
            self.data[m], S, V = truncate_A(A, S, V, D)
            if m+1 < len(self.data):
                self.data[m+1] = swapaxes(tensordot(dot(S, V),
                                          self.data[m+1], (-1, 1)), 0, 1)

        Ls = [None]*self.L
        Vs = [None]*self.L
        V = S = identity(1)

        for m in range(len(self.data))[::-1]:
            # sort out env canonicalisation
            M = trace(dot(dot(dot(dot(self.data[m],
                                  V),
                              S),
                          H(V)),
                      H(self.data[m])), 0, 0, 2)

            Vs[m], Ls[m] = diagonalise(M)

            self.data[m] = swapaxes(dot(dot(H(Vs[m]), self.data[m]), V),
                                    0, 1)

            V, S = Vs[m], Ls[m]

        Ls.append(ones_like(Ls[0]))  # Ones on each end

        for m in range(len(self.data)):
            # sort out the rank (tDj is tilde)
            Dj = Ls[m].shape[0]
            tDj = rank(Ls[m])
            P = concatenate([identity(tDj), zeros((tDj, Dj-tDj))], axis=1)
            self.data[m-1] = self.data[m-1] @ H(P)
            self.data[m] = P @ self.data[m]
            Ls[m] = P @ Ls[m] @ H(P)

        self.Ls = Ls  # store all the singular values

        if testing:
            self.update_properties()
            self.ok = self.ok\
                and self.is_left_canonical\
                and self.is_left_env_canonical\
                and self.is_full_rank\
                and self.has_trace_1

        return self

    def mixed_canonicalise(self, oc, D=None, testing=True):
        """mixed_canonicalise: bring internal fMPS to mixed canonical form with
        orthogonality center oc, potentially with a truncation

        :param oc: orthogonality center
        :param D: bond dimension
        """
        self.ok = True
        self.oc = oc
        self.right_canonicalise(D)

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
                self.data[m+1] = swapaxes(tensordot(dot(S, V),
                                                    self.data[m+1],
                                                    (-1, 1)), 0, 1)

        if testing:
            self.ok = self.ok and is_left_canonical(self.data[:oc])\
                              and is_right_canonical(self.data[oc+1:])

        return self

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

    def overlap(self, other, pad_other=True):
        """overlap

        :param other: other with which to calculate overlap
        :param pad_other: whether to try to pad the other
        other if they are different sizes, or to raise an error
        """
        assert len(self) == len(other)

        padded_self, padded_other = mps_pad(self, other)

        F = ones((1, 1))
        for L, R in zip(padded_self, padded_other):
            F = tensordot(tensordot(H(L), F, (-1, 1)), R, ([0, -1], [0, 1]))
        return F[0][0]

    def norm(self):
        """norm: not efficient - computes full overlap. 
        use self.E(identity(self.d), site) for mixed 
        canonicalising version"""
        return self.overlap(self)

    def E(self, op, site):
        M = self.mixed_canonicalise(site)[site]
        return tensordot(op, trace(dot(H(M),  M), axis1=1, axis2=3), [[0, 1], [0, 1]])
         
class vfMPS(object):
    """vidal finite MPS
    lists of tuples (\Gamma, \Lambda) of numpy arrays"""
    def __init__(self,  data=None, d=None):
        """__init__

        :param data: matrices in form [(\Gamma, \Lambda)]
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
                S = [(sigmax().full(), site1), (sigmay().full(), site2), (sigmaz().full(), site3)]
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
                S = [(sigmax().full(), site1), (sigmay().full(), site2), (sigmaz().full(), site3)]
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
    unittest.main(verbosity=1)
