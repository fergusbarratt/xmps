from xmps.fMPS import fMPS, vfMPS, fTFD, fs

import unittest

import numpy as np
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
from numpy import linspace

from scipy.linalg import null_space as null, orth, expm#, sqrtm as ch
from scipy.linalg import polar, block_diag as bd
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from xmps.tests import is_right_canonical, is_right_env_canonical, is_full_rank
from xmps.tests import is_left_canonical, is_left_env_canonical, has_trace_1

from xmps.tensor import H as cT, truncate_A, truncate_B, diagonalise, rank, mps_pad
from xmps.tensor import C as c, lanczos_expm, tr_svd, T
from xmps.tensor import rdot, ldot, structure
from xmps.fMPS import lt_ as lt_

from xmps.spin import n_body, N_body_spins, spins

from copy import deepcopy, copy
from functools import reduce
from itertools import product
import cProfile
from time import time
import uuid

Sx, Sy, Sz = spins(0.5)
Sx, Sy, Sz = 2*Sx, 2*Sy, 2*Sz

from xmps.ncon import ncon as ncon

class TestfMPS(unittest.TestCase):
    """TestfMPS"""

    def setUp(self):
        """setUp"""
        self.N = N = 5  # Number of MPSs to test
        #  min and max params for randint
        L_min, L_max = 7, 8
        d_min, d_max = 2, 3
        D_min, D_max = 9, 10
        ut_min, ut_max = 3, 7
        # N random MPSs
        self.rand_cases = [fMPS().random(randint(L_min, L_max),
                                         randint(d_min, d_max),
                                         randint(D_min, D_max))
                           for _ in range(N)]
        # N random MPSs, right canonicalised and truncated to random D
        self.right_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, d_max),
                            randint(D_min, D_max)).right_canonicalise(
                            randint(D_min, D_max))
                            for _ in range(N)]
        self.right_ortho_cases = [fMPS().random(
                                randint(L_min, L_max),
                                randint(d_min, d_max),
                                randint(D_min, D_max)).right_orthogonalise()
                           for _ in range(N)]
        # N random MPSs, left canonicalised and truncated to random D
        self.left_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, d_max),
                            randint(D_min, D_max)).left_canonicalise(
                            randint(D_min, D_max))
                           for _ in range(N)]
        self.left_ortho_cases = [fMPS().random(
                                randint(L_min, L_max),
                                randint(d_min, d_max),
                                randint(D_min, D_max)).left_orthogonalise()
                           for _ in range(N)]
        self.mixed_cases = [fMPS().random(
                            randint(L_min, L_max),
                            randint(d_min, d_max),
                            randint(D_min, D_max)).mixed_canonicalise(
                            randint(ut_min, ut_max),
                            randint(D_min, D_max))
                            for _ in range(N)]


        # finite fixtures 
        fix_loc = 'fixtures/'
        self.tens_0_2 = load(fix_loc+'mat2x2.npy')
        self.tens_0_3 = load(fix_loc+'mat3x3.npy')
        self.tens_0_4 = load(fix_loc+'mat4x4.npy')
        self.tens_0_5 = load(fix_loc+'mat5x5.npy')
        self.tens_0_6 = load(fix_loc+'mat6x6.npy')
        self.tens_0_7 = load(fix_loc+'mat7x7.npy')
        self.tens_0_8 = load(fix_loc+'mat8x8.npy')
        self.tens_0_9 = load(fix_loc+'mat9x9.npy')

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

        self.mps_0_8 = fMPS().left_from_state(self.tens_0_8)
        self.psi_0_8 = self.mps_0_8.recombine().reshape(-1)

        self.mps_0_9 = fMPS().left_from_state(self.tens_0_9)
        self.psi_0_9 = self.mps_0_9.recombine().reshape(-1)
        self.fixtures = [self.mps_0_2, self.mps_0_3, self.mps_0_4,
                         self.mps_0_5, self.mps_0_6, self.mps_0_7,
                         self.mps_0_8, self.mps_0_9]

        self.all_cases = self.rand_cases+self.right_cases+self.left_cases+self.mixed_cases+self.fixtures

    def test_orthogonalise(self):
        for case_ in self.rand_cases:
            case = case_.copy().left_orthogonalise()
            self.assertTrue(is_left_canonical(case.data))
            self.assertTrue(not allclose(case.norm_, 1))
            self.assertTrue(allclose(case.norm(), 1))

            case = case_.copy().right_orthogonalise()
            self.assertTrue(is_right_canonical(case.data))
            self.assertTrue(not allclose(case.norm_, 1))
            self.assertTrue(allclose(case.norm(), 1))

            old_evs = case.Es([Sx, Sy, Sz], 3)
            for _ in range(10):
                case = case.right_orthogonalise()
                self.assertTrue(np.allclose(old_evs, case.Es([Sx, Sy, Sz], 3)))

                case = case.left_orthogonalise()
                self.assertTrue(np.allclose(old_evs, case.Es([Sx, Sy, Sz], 3)))

            for i in range(case_.L):
                case = case_.copy().mixed_orthogonalise(i)
                self.assertTrue(is_left_canonical(case.data[:i]))
                self.assertTrue(is_right_canonical(case.data[i+1:]))

                self.assertTrue(np.allclose(case_.copy().E(Sx, i), case_.copy().E_(Sx, i)))
                self.assertTrue(np.allclose(case_.copy().E(Sy, i), case_.copy().E_(Sy, i)))
                self.assertTrue(np.allclose(case_.copy().E(Sz, i), case_.copy().E_(Sz, i)))

    def test_overlap(self):
        for case1, case2 in product(self.left_cases[1:], self.left_cases):
            self.assertAlmostEqual(np.abs(case1.overlap(case2)), np.abs(case1.full_overlap(case2)))

    def test_random_with_energy(self):
        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        L = 6
        d = 2
        D = 1

        h = 4*Sz12@Sz22+2*Sx22+2*Sz22
        h = (h+h.conj().T)/2
        H = [h for _ in range(L-1)]
        H[0] = H[0]+2*Sx12+2*Sz12

        comH = sum([n_body(a, i, len(H), d=2) for i, a in enumerate(H)], axis=0)
        v = eigvalsh(comH)
        E = (max(v)-min(v))/2.5

        tol = 1e-10
        mps = fMPS().random_with_energy_E(E, H, L, d, D, 1e-10)
        self.assertTrue(abs(mps.energy(H)-E)<tol)

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
        from numpy import nonzero, logical_not as lon, max
        for m, L in enumerate(range(2, 8)):
            H = [Sz1@Sz2+(Sz1+Sz2) for _ in range(L-1)]
            mps = self.fixtures[m].right_canonicalise(1).grow(H, 0.1, 2**L)
            mps.right_canonicalise()
            #mps = fMPS().random(L, 2, 2**L).right_canonicalise()
            dA_dt = mps.dA_dt(H)
            l, r = mps.get_envs()

            def inv_diag(mat):
                return diag(1/diag(mat))
            def ch_diag(mat):
                return diag(sqrt(diag(mat)))
            cr, cl = [ch_diag(r(i)) for i in range(mps.L)], [ch_diag(l(i)) for i in range(mps.L)]
            icr, icl = [inv_diag(cr[i]) for i in range(mps.L)], [inv_diag(cl[i]) for i in range(mps.L)]
            inv_envs= (icr, icl, cr, cl)

            for i, j in product(range(L), range(L)):
                F2 = -1j*mps.F2(i, j, H, inv_envs=inv_envs)
                Γ2 = mps.christoffel(i, j, min(i, j), closed=(None, None, l(min(i, j)-1)@dA_dt[min(i, j)]@r(min(i, j))))
                self.assertTrue(allclose(F2, -Γ2))

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
        from xmps.mps_examples import comp_z
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
        J1, J2, F = mps.jac(H, True, False)
        J2 = J2+F
        self.assertTrue(allclose(J1, -1j*eye(3)))
        self.assertTrue(allclose(J2, 0))
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J, kron(1j*Sy, eye(3))))

        mps = self.mps_0_3
        H = [eye(4)/2, eye(4)/2]
        J1, J2, F = mps.jac(H, True, False)
        J2 = J2+F
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J1, -1j*eye(7)))
        self.assertTrue(allclose(J2, 0))
        self.assertTrue(allclose(J, kron(1j*Sy, eye(7))))

        mps = self.mps_0_4
        H = [eye(4)/3, eye(4)/3, eye(4)/3]
        J1, J2, F = mps.jac(H, True, False)
        J2 = J2+F
        J = mps.jac(H, True, True)
        self.assertTrue(allclose(J1, -1j*eye(15)))
        self.assertTrue(allclose(J2, 0))
        self.assertTrue(allclose(J, kron(1j*Sy, eye(15))))

    def test_jac_no_projection(self):
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

        mps = self.mps_0_3.left_canonicalise()
        H = [Sz1@Sz2+Sx1, Sz1@Sz2+Sx1+Sx2]
        J1, J2, F = mps.jac(H, True, False)
        J2 = J2+F
        self.assertTrue(allclose(J1+J1.conj().T, 0))
        from numpy import nonzero
        for L in range(2, 7):
            mps = fMPS().random(L, 2, 2**L).left_canonicalise()
            N = 10
            for _ in range(N):
                H = [randn(4, 4)+1j*randn(4, 4) for _ in range(L-1)]
                H = [h+h.conj().T for h in H]
                J1, J2, F = mps.jac(H, True, False)
                J2 = J2+F
                self.assertTrue(allclose(J1,-J1.conj().T))
                J2[abs(J2)<1e-10]=0
                if not allclose(J2, 0):
                    print(J2[nonzero(J2)])
                self.assertTrue(allclose(J2, 0))

                #J = mps.jac(H)
                #self.assertTrue(allclose(J+J.T, 0))

                mps = (mps+mps.dA_dt(H)*0.1).left_canonicalise()

    def test_expand(self):
        mps = fMPS().random(3, 2, 1)
        self.assertTrue(mps.structure()==[(1, 1), (1, 1), (1, 1)])
        ls = mps.Es([Sx, Sy, Sz], 0)

        mps.expand(2)
        l, r = mps.get_envs()

        self.assertTrue(allclose(ls, mps.Es([Sx, Sy, Sz], 0)))
        self.assertTrue(mps.structure()==[(1, 2), (2, 2), (2, 1)])

    def test_match_gauge_to(self):
        Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
        Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

        mps_1 = self.right_cases[0]
        # doesn't change expectation values
        for case in self.right_cases:
            a = mps_1.E(Sx, 0)
            mps_1.right_canonicalise().match_gauge_to(case)
            b = mps_1.E(Sx, 0)
            self.assertTrue(allclose(a, b))

        mps_1 = self.left_cases[0]
        for case in self.left_cases:
            a = mps_1.E(Sx, 0)
            mps_1.left_canonicalise().match_gauge_to(case)
            b = mps_1.E(Sx, 0)
            self.assertTrue(allclose(a, b))

        mps_1 = self.left_cases[0]
        for case in self.right_cases:
            a = mps_1.E(Sx, 0)
            mps_1.left_canonicalise().match_gauge_to(case)
            b = mps_1.E(Sx, 0)
            self.assertTrue(allclose(a, b))

        evs = []
        evs_ = []
        mps_0 = fMPS().random(6, 2, 5)
        H = [Sz12@Sz22+Sx12+Sx22, Sz12@Sz22+Sx22, Sz12@Sz22+Sx22+Sx22, Sz12@Sz22+Sx22, Sz12@Sz22+Sx22]

        mps = mps_0.copy()
        for _ in range(30):
            evs.append(mps.E_(Sx, 0))
            old_mps = mps.copy()
            mps = (mps + mps.dA_dt(H)*0.1).left_canonicalise()
            mps = mps.match_gauge_to(old_mps).copy()

        mps = mps_0.copy()
        for _ in range(30):
            evs_.append(mps.E_(Sx, 0))
            mps = (mps + mps.dA_dt(H)*0.1).left_canonicalise()

        self.assertTrue(allclose(array(evs),array(evs_)))

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

        psi_0_2 = load('fixtures/mat2x2.npy')
        self.tfd_0_2 = fTFD().from_fMPS(fMPS().left_from_state(psi_0_2))

    def test_symmetries(self):
        """test_symmetry"""
        for A, A_ in zip(self.pure_cases, self.mixed_cases):
            self.assertTrue(A==A.symmetry())
            self.assertFalse(A_==A_.symmetry())
            for n in range(A.L):
                vL_sym, vL_asy, M = A.get_vL()[n]
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

                vL_sym, vL_asy, M = A_.get_vL()[n]
                self.assertTrue(allclose(vL_sym,  M@fs(vL_sym)))
                self.assertTrue(allclose(vL_asy, -M@fs(vL_asy)))

    def test_time_evolution(self):
        """test_time_evolution"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        H = [Sz1@Sz2+Sx1+Sx2]
        T = linspace(0, 0.1, 2)
        A = self.tfd_0_2
        evs = []
        for _ in range(400):
            evs.append(A.Es([Sx, Sy, Sz], 1))
            A = (A+A.dA_dt(H)*0.1).left_canonicalise()

if __name__ == '__main__':
    unittest.main(verbosity=2)
