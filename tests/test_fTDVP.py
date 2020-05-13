import os
import glob
import shutil
import unittest
import pickle

from time import time
from xmps.fMPS import fMPS
from xmps.tensor import get_null_space, H as cT, C as c, partial_trace
from xmps.ncon import ncon
from xmps.spin import N_body_spins, spins, comm, n_body

from xmps.fTDVP import Trajectory
try:
    from xmps.tdvp.tdvp_fast import tdvp, MPO_TFI
except ModuleNotFoundError:
    tdvp_available = False
else:
    tdvp_available = True

from numpy import array, linspace, real as re, reshape, sum, swapaxes as sw
from numpy import tensordot as td, squeeze, trace as tr, expand_dims as ed
from numpy import load, isclose, allclose, zeros_like as zl, prod, imag as im
from numpy import log, abs, diag, cumsum as cs, arange as ar, eye, kron as kr
from numpy import cross, dot, kron, split, concatenate as ct, isnan, isinf
from numpy import trace as tr, zeros, printoptions, tensordot, trace, save
from numpy import sign, block, sqrt, max, sort
from numpy.random import randn
from numpy.linalg import inv, svd, eig, eigvalsh
from numpy.linalg import det, qr
import numpy as np

from scipy.linalg import sqrtm, expm, norm, null_space as null, cholesky as ch
from scipy.sparse.linalg import expm_multiply, expm

from matplotlib import pyplot as plt
from functools import reduce
from copy import copy, deepcopy
from tqdm import tqdm
Sx, Sy, Sz = spins(0.5)
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

class TestTrajectory(unittest.TestCase):
    """TestF"""
    def setUp(self):
        """setUp"""
        fix_dir = 'fixtures/'
        self.tens_0_2 = load(fix_dir+'mat2x2.npy')
        self.tens_0_3 = load(fix_dir+'mat3x3.npy')
        self.tens_0_4 = load(fix_dir+'mat4x4.npy')
        self.tens_0_5 = load(fix_dir+'mat5x5.npy')

        self.mps_0_1 = fMPS().left_from_state(self.tens_0_2).right_canonicalise(1)

        self.mps_0_2 = fMPS().left_from_state(self.tens_0_2)
        self.psi_0_2 = self.mps_0_2.recombine().reshape(-1)

        self.mps_0_3 = fMPS().left_from_state(self.tens_0_3)
        self.psi_0_3 = self.mps_0_3.recombine().reshape(-1)

        self.mps_0_4 = fMPS().left_from_state(self.tens_0_4)
        self.psi_0_4 = self.mps_0_4.recombine().reshape(-1)

        self.mps_0_5 = fMPS().left_from_state(self.tens_0_5)
        self.psi_0_5 = self.mps_0_5.recombine().reshape(-1)

    @unittest.skipIf(not tdvp_available, 'tdvp module not available')
    def test_stop_resume(self):
        """test_stop_resume"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

        dt, t_fin = 2e-2,5
        T = linspace(0, t_fin, int(t_fin//dt)+1)

        mps_0 = self.mps_0_4.right_canonicalise(3)
        H = [Sz1@Sz2+Sx1+Sx2] + [Sz1@Sz2+Sx2] + [Sz1@Sz2+Sx2]
        W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0)]
        F = Trajectory(mps_0, H=H, W=W)
        F.run_name = 'test'

        F.lyapunov(T, t_burn=0)
        F.stop()
        F_ = Trajectory().resume('test')
        self.assertTrue(F_.mps_0==F.mps_0)
        self.assertTrue(F_.mps==F.mps)
        self.assertTrue(allclose(F_.Q, F.Q))
        self.assertTrue(allclose(F_.lys, F.lys))
        self.assertTrue(all([allclose(F.vL[i], F_.mps.old_vL[i]) for i in range(len(F.vL))]))
        F_.lyapunov(T)
        self.assertTrue(len(F_.lys)==2*len(T)-1)

        plt.plot(F.vs+F_.vs)
        plt.show()

        plt.plot(F_.lys)
        plt.show()

        plt.plot(F.mps_history+F_.mps_history)
        plt.show()

        shutil.rmtree(F_.run_dir)

    @unittest.skipIf(not tdvp_available, 'tdvp module not available')
    def test_integrators(self):
        test_D_1 = False
        if test_D_1:
            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

            dt, t_fin = 2e-2, 10
            T = linspace(0, t_fin, int(t_fin//dt)+1)

            mps_0 = self.mps_0_2.left_canonicalise(1)
            H = [Sz1@Sz2+Sx1+Sx2]
            W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0)]
            F = Trajectory(mps_0, H=H, W=W)

            C = F.invfreeint(T).mps_evs([Sx, Sy, Sz], 0)
            F.clear()
            D = F.eulerint(T).mps_evs([Sx, Sy, Sz], 0)
            F.clear()

            plot = True
            if plot:
                plt.plot(T, C)
                plt.plot(T, D)
                plt.show()

        test_2 = False
        if test_2:
            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

            dt, t_fin = 2e-2, 10
            T = linspace(0, t_fin, int(t_fin//dt)+1)

            mps_0 = self.mps_0_2
            H = [Sz1@Sz2+Sx1+Sx2]
            W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0)]
            F = Trajectory(mps_0, H=H, W=W)

            A = F.edint(T).ed_evs([Sx1, Sy1, Sz1])
            F.clear()
            B = F.eulerint(T).mps_evs([Sx, Sy, Sz], 0)
            F.clear()
            C = F.invfreeint(T).mps_evs([Sx, Sy, Sz], 0)
            F.clear()
            self.assertTrue(norm(B-A)/prod(A.shape)<dt**2)
            self.assertTrue(norm(C-A)/prod(A.shape)<dt**2)

            plot=False
            if plot:
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
                ax.set_ylim([-1, 1])
                ax.plot(T, A)
                ax.plot(T, B)
                ax.plot(T, C)
                plt.show()

        test_3 = False
        if test_3:
            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

            dt, t_fin = 2e-2, 10
            T = linspace(0, t_fin, int(t_fin//dt)+1)

            mps_0 = self.mps_0_3.right_canonicalise()
            H = [Sz1@Sz2+Sx1+Sx2] + [Sz1@Sz2+Sx2]
            W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0)]
            F = Trajectory(mps_0, H=H, W=W)

            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 3)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 3)
            Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 3)


            A = F.edint(T).ed_evs([Sx3, Sy3, Sz3])
            F.clear()
            B = F.eulerint(T).mps_evs([Sx, Sy, Sz], 2)
            F.clear()
            C = F.invfreeint(T).mps_evs([Sx, Sy, Sz], 2)
            self.assertTrue(norm(B-A)/prod(A.shape)<dt**2)
            self.assertTrue(norm(C-A)/prod(A.shape)<dt**2)

            plot=False
            if plot:
                fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
                ax.set_ylim([-1, 1])
                ax.plot(T, A)
                ax.plot(T, B)
                ax.plot(T, C)
                plt.show()

        test_4 = True
        if test_4:
            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

            dt, t_fin = 0.5, 10
            T = linspace(0, t_fin, int(t_fin//dt)+1)

            mps_0 = self.mps_0_4.right_canonicalise(1)
            H = [Sz1@Sz2+Sx1+Sx2+Sz1+Sz2] + [Sz1@Sz2+Sx2+Sz2] + [Sz1@Sz2+Sx2+Sz2]
            W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0.5)]
            F = Trajectory(mps_0, H=H, W=W)

            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
            Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
            Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)

            X = F.edint(T)
            A = X.ed_evs([Sx3, Sy3, Sz3])
            A_e = X.ed_energies()
            F.clear()

            X = F.eulerint(T)
            B = X.mps_evs([Sx, Sy, Sz], 2)
            X.ed_history = array([mps.recombine().reshape(-1) for mps in X.mps_list()])
            B_e = X.ed_energies()
            F.clear()

            X = F.invfreeint(T)
            C = X.mps_evs([Sx, Sy, Sz], 2)
            X.ed_history = array([mps.recombine().reshape(-1) for mps in X.mps_list()])
            C_e = X.ed_energies()

            #self.assertTrue(norm(B-A)/prod(A.shape)<dt)
            #self.assertTrue(norm(C-A)/prod(A.shape)<dt**2)

            plot=True
            if plot:
                fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
                ax.set_ylim([-1, 1])
                ax.plot(T, A)
                ax.plot(T, B)
                ax.plot(T, C)
                plt.show()
                plt.plot(T, A_e, label='ed')
                plt.plot(T, B_e, label='euler')
                plt.plot(T, C_e, label='invfree')
                plt.legend()
                plt.show()

        test_5 = False
        if test_5:
            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

            dt, t_fin = 2e-2, 10
            T = linspace(0, t_fin, int(t_fin//dt)+1)

            mps_0 = self.mps_0_5.right_canonicalise()
            H = [Sz1@Sz2+Sx1+Sx2] + [Sz1@Sz2+Sx2] + [Sz1@Sz2+Sx2] + [Sz1@Sz2+Sx2]
            W = mps_0.L*[MPO_TFI(0, 0.25, 0.5, 0)]
            F = Trajectory(mps_0, H=H, W=W)

            Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 5)
            Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 5)
            Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 5)
            Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 5)
            Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 5)

            X = F.invfreeint(T, 'low')
            C = X.mps_evs([Sx, Sy, Sz], 1)
            C_e = X.mps_energies()
            X.ed_history = array([mps.recombine().reshape(-1) for mps in X.mps_list()])
            C_e_ = X.ed_energies()
            F.clear()

            X = F.edint(T)
            A = X.ed_evs([Sx2, Sy2, Sz2])
            A_e = X.ed_energies()
            F.clear()

            X = F.eulerint(T)
            B = X.mps_evs([Sx, Sy, Sz], 1)
            B_e = X.mps_energies()
            X.ed_history = [mps.recombine().reshape(-1) for mps in X.mps_list()]
            B_e_ = X.ed_energies()
            F.clear()

            self.assertTrue(norm(B-A)/prod(A.shape)<dt)
            self.assertTrue(norm(C-A)/prod(A.shape)<dt**2)

            plot=False
            if plot:
                fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
                ax.set_ylim([-1, 1])
                ax.plot(T, A)
                #ax.plot(T, B)
                ax.plot(T, C)
                plt.show()
                fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
                ax.plot(T, A_e, label='ed')
                #ax.plot(T, B_e, label='euler')
                #ax.plot(T, B_e_, label='euler_')
                #ax.plot(T, C_e, label='invfree')
                ax.plot(T, C_e_, label='invfree_')
                plt.legend()
                plt.show()

    @unittest.skipIf(not tdvp_available, 'tdvp module not available')
    def test_OTOCs_eye(self):
        """OTOC zero for W=eye"""
        Sx1,  Sy1,  Sz1 =  N_body_spins(0.5, 1,  5)
        Sx2,  Sy2,  Sz2 =  N_body_spins(0.5, 2,  5)
        Sx3,  Sy3,  Sz3 =  N_body_spins(0.5, 3,  5)
        Sx4,  Sy4,  Sz4 =  N_body_spins(0.5, 4,  5)
        Sx5,  Sy5,  Sz5 =  N_body_spins(0.5, 5,  5)

        mps = fMPS().random(5, 2, None).left_canonicalise()
        W = mps.L*[MPO_TFI(0, 0.25, 0.5, 0.5)]
        H = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4+Sz4@Sz5+\
                Sx1+Sx2+Sx3+Sx4+Sx5+\
                Sz1+Sz2+Sz3+Sz4+Sz5
        ops = eye(2**5), eye(2**5)
        ops_ = ((eye(2), 0), (eye(2), 0))
        T = linspace(0, 1, 3)
        F = Trajectory(mps, H=H, W=W)
        ed_evs  = F.ed_OTOC(T, ops)
        mps_evs = F.mps_OTOC(T, ops_)
        plt.plot(mps_evs)
        plt.show()
        self.assertTrue(allclose(ed_evs, 0))
        self.assertTrue(allclose(mps_evs, 0))

    @unittest.skipIf(not tdvp_available, 'tdvp module not available')
    def test_OTOCs_ed(self):
        """test mps otocs against ed"""
        Sx1,  Sy1,  Sz1 =  N_body_spins(0.5, 1,  5)
        Sx2,  Sy2,  Sz2 =  N_body_spins(0.5, 2,  5)
        Sx3,  Sy3,  Sz3 =  N_body_spins(0.5, 3,  5)
        Sx4,  Sy4,  Sz4 =  N_body_spins(0.5, 4,  5)
        Sx5,  Sy5,  Sz5 =  N_body_spins(0.5, 5,  5)

        mps = fMPS().random(5, 2, None).left_canonicalise()
        W = mps.L*[MPO_TFI(0, 0.25, 0.5, 0.5)]
        H = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4+Sz4@Sz5+\
                Sx1+Sx2+Sx3+Sx4+Sx5+\
                Sz1+Sz2+Sz3+Sz4+Sz5
        ops = Sz2, Sz5
        ops_ = ((Sz, 1), (Sz, 4))
        T = linspace(0, 1, 3)
        F = Trajectory(mps, H=H, W=W, fullH=True)
        ed_evs  = F.ed_OTOC(T, ops)
        mps_evs = F.mps_OTOC(T, ops_)
        plt.plot(mps_evs)
        plt.plot(ed_evs)
        plt.show()
        self.assertTrue(allclose(ed_evs, 0))
        self.assertTrue(allclose(mps_evs, 0))

    def test_lyapunov_local_hamiltonian_2(self):
        """zero lyapunov exponents with local hamiltonian"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_2
        H = [Sx1-Sx2]
        T = linspace(0, 1, 100)
        exps, lys, _ = Trajectory(mps, H).lyapunov(T, D=1)
        self.assertTrue(allclose(exps[-1], 0))

    def test_lyapunov_local_hamiltonian_3(self):
        """zero lyapunov exponents with local hamiltonian"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_3
        H = [Sx1+Sx2, Sx2]
        T = linspace(0, 1, 100)
        exps, lys, _ = Trajectory(mps, H).lyapunov(T, D=1)
        self.assertTrue(allclose(exps[-1], 0))

    def test_lyapunov_no_truncate_2(self):
        """zero lyapunov exponents with no projection"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_2
        H = [Sz1@Sz2+Sx1+Sx2]
        T = linspace(0, 1, 100)
        exps, lys, _ = Trajectory(mps, H).lyapunov(T, D=None)
        self.assertTrue(allclose(exps[-1], 0))

    def test_lyapunov_no_truncate_3(self):
        """zero lyapunov exponents with no truncation"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps = self.mps_0_3
        H = [Sz1@Sz2+Sx1+Sx2, Sz1@Sz2+Sx2]
        T = linspace(0, 0.1, 100)
        exps, lys, _ = Trajectory(mps, H).lyapunov(T, D=None)
        self.assertTrue(allclose(exps[-1], 0))

if __name__ == '__main__':
    unittest.main(verbosity=2)
