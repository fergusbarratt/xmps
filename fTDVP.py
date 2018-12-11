import os
import glob
import shutil
import unittest
import pickle

from time import time
from fMPS import fMPS
from tensor import get_null_space, H as cT, C as c
from ncon import ncon
from tdvp.tdvp_fast import tdvp, MPO_TFI

from spin import N_body_spins, spins, comm, n_body
from numpy import array, linspace, real as re, reshape, sum, swapaxes as sw
from numpy import tensordot as td, squeeze, trace as tr, expand_dims as ed
from numpy import load, isclose, allclose, zeros_like as zl, prod, imag as im
from numpy import log, abs, diag, cumsum as cs, arange as ar, eye, kron as kr
from numpy import cross, dot, kron, split, concatenate as ct, isnan, isinf
from numpy import trace as tr, zeros, printoptions, tensordot, trace, save
from numpy import sign, block, sqrt, max
from numpy.random import randn
from numpy.linalg import inv, svd, eig
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

class Trajectory(object):
    """Trajectory"""

    def __init__(self, mps_0=None, H=None, W=None, fullH=False, run_name='', continuous=True):
        """__init__

        :param mps_0: initial state
        :param H: hamiltonian
        :param W: mpo (for invfreeint)
        :param T: time steps
        :param run_name: prefix for saving
        """
        self.H = H # hamiltonian as list of 4x4 mats or big matrix
        self.W = W # hamiltonian as mpo - required for invfreeint

        self.mps_0 = mps_0.copy() if mps_0 is not None else mps_0
        self.mps = mps_0.copy() if mps_0 is not None else mps_0
        self.fullH=fullH
        self.mps_history = []
        self.run_name = run_name
        self.continuous = continuous

    def euler(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
        return (mps + mps.dA_dt(H, fullH=self.fullH)*dt).left_canonicalise()

    def rk4(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
        k1 = mps.dA_dt(H, fullH=self.fullH)*dt
        k2 = (mps+k1/2).dA_dt(H, fullH=self.fullH)*dt
        k3 = (mps+k2/2).dA_dt(H, fullH=self.fullH)*dt
        k4 = (mps+k3).dA_dt(H, fullH=self.fullH)*dt

        return (mps+(k1+2*k2+2*k3+k4)/6).left_canonicalise()

    def invfree(self, mps, dt, W=None, store=True):
        if store:
            self.mps_history.append(mps.serialize(real=True))
        if self.continuous:
            A_old = mps.copy()
        A = tdvp(mps.data, self.W, 1j*dt/2)
        return fMPS(A[0]) if not self.continuous else fMPS(A[0]).match_gauge_to(A_old)

    def invfree4(self, mps, dt, W=None, store=True):
        """invfree4 fourth order symmetric composition
        """
        if store:
            self.mps_history.append(mps.serialize(real=True))
        if self.continuous:
            A_old = mps.copy()
        a1, a2, a3 = b5, b4, b3 = (146+5*sqrt(19))/540, (-2+10*sqrt(19))/135, 1/5
        b1, b2 = a5, a4 = (14-sqrt(19))/108, (-23-20*sqrt(19))/270
        A = tdvp(tdvp(tdvp(tdvp(tdvp(mps.data,
                                     self.W, 1j*dt, a=a1, b=b1)[0],
                                 self.W, 1j*dt, a=a2, b=b2)[0],
                             self.W, 1j*dt, a=a3, b=b3)[0],
                       self.W, 1j*dt, a=a4, b=b4)[0],
                   self.W, 1j*dt, a=a5, b=b5)
        return fMPS(A[0]) if not self.continuous else fMPS(A[0]).match_gauge_to(A_old)

    def eulerint(self, T):
        """eulerint: integrate with euler time steps

        :param T:
        """
        mps, H = self.mps.left_canonicalise(), self.H
        L, d, D = mps.L, mps.d, mps.D

        for t in tqdm(T):
            mps = self.euler(mps, T[1]-T[0])

        self.mps = fMPS().deserialize(self.mps_history[-1], L, d, D, real=True)
        return self

    def rk4int(self, T):
        """rk4int: integrate with rk4 timesteps
        """
        mps, H = self.mps.left_canonicalise(), self.H
        L, d, D = mps.L, mps.d, mps.D

        for t in tqdm(T):
            mps = self.rk4(mps, T[1]-T[0])

        self.mps = fMPS().deserialize(self.mps_history[-1], L, d, D, real=True)
        return self

    def invfreeint(self, T, D=None, order='low'):
        """invfreeint: inverse free (symmetric) integrator - wraps Frank code.
                       requires a MPO - MPO_TFI/MPO_XXZ from tdvp
                       remember my ed thing has Sx = 1/2 Sx etc. for some reason

        :param T:
        :param D:
        """
        assert self.W is not None
        mps, H = self.mps, self.H
        L, d, D = mps.L, mps.d, mps.D

        for t in tqdm(T):
            if order=='high':
                mps = self.invfree4(mps, T[1]-T[0])
            elif order=='low':
                mps = self.invfree(mps, T[1]-T[0])

        self.mps = fMPS().deserialize(self.mps_history[-1], L, d, D, real=True)
        return self

    def odeint(self, T):
        """odeint: integrate TDVP equations with scipy.odeint
           bar is upper bound - might do fewer iterations than it expects.
           Use another method for more predictable results

        :param T: timesteps
        :param D: bond dimension to truncate initial state to
        """
        mps, H = self.mps.left_canonicalise(), self.H
        L, d, D = mps.L, mps.d, mps.D
        bar = tqdm()
        m=0

        def f(t, v):
            """f_odeint: f acting on real vector

            :param v: Vector: [reals, imags]
            :param t: Time
            """
            bar.update()
            nonlocal m
            m+=1
            if m==4:
                return fMPS().deserialize(v, L, d, D, real=True)\
                            .left_canonicalise()\
                            .dA_dt(H, fullH=self.fullH)\
                            .serialize(real=True)
            else:
                return fMPS().deserialize(v, L, d, D, real=True)\
                            .dA_dt(H, fullH=self.fullH)\
                            .serialize(real=True)


        v = mps.serialize(real=True)
        Q = solve_ivp(f, (T[0], T[-1]), v, method='LSODA', t_eval=T,
                      max_step=T[1]-T[0], min_step=T[1]-T[0])
        traj = Q.y.T
        bar.close()

        self.mps_history = traj
        self.mps = fMPS().deserialize(traj[-1], L, d, D, real=True)
        return self

    def edint(self, T):
        """edint: exact diagonalisation
        """
        H = self.H
        psi_0 = self.mps.recombine().reshape(-1)
        H = sum([n_body(a, i, len(H), d=2)
                 for i, a in enumerate(H)], axis=0) if not self.fullH else H
        psi_n = psi_0
        self.ed_history = [psi_0]
        dt = T[1]-T[0]
        for t in tqdm(T[:-1]):
            psi_n = expm(-1j *H*dt)@psi_n
            self.ed_history.append(psi_n)

        self.ed_history = array(self.ed_history)
        self.psi = self.ed_history[-1]
        self.mps = fMPS().left_from_state(self.psi.reshape([self.mps.d]*self.mps.L))
        return self

    def lyapunov(self, T, D=None,
                 just_max=False,
                 t_burn=2,
                 initial_basis='F2', 
                 order='high'):
        self.has_run_lyapunov = True
        H = self.H
        has_mpo = self.W is not None
        if D is not None and t_burn!=0 and not hasattr(self, 'Q'):
            # if MPO supplied - just expand, canonicalise and use inverse free integrator
            # otherwise use dynamical expand: less numerically stable
            # if we already have a basis set - we must be resuming a run
            if has_mpo:
                self.mps = self.mps.right_canonicalise().expand(D)
                self.invfreeint(linspace(0, t_burn, int(50*t_burn)), order='high')
                self.burn_len = int(200*t_burn)
                self.mps_history = []
            else:
                self.mps = self.mps.grow(self.H, 0.1, D).right_canonicalise()
                self.rk4int(linspace(0, 1, 100))

        if hasattr(self, 'Q'):
            Q = self.Q
        elif initial_basis == 'F2':
            Q = self.mps.tangent_space_basis(H=H, type=initial_basis)
        elif initial_basis == 'eye' or initial_basis == 'rand':
            Q = kron(eye(2), self.mps.tangent_space_basis(H=H, type=initial_basis))
        else:
            Q = initial_basis

        if just_max:
            # just evolve top vector, dont bother with QR
            q = Q[0]
        dt = T[1]-T[0]
        lys = []
        self.vs = []
        for t in tqdm(range(len(T))):
            J = self.mps.jac(H)
            if hasattr(self.mps, 'old_vL'):
                self.vs.append(self.mps.v)
            if just_max:
                q = expm_multiply(J*dt, q)
                lys.append(log(abs(norm(q))))
                q /= norm(q)
            else:
                M = expm_multiply(J*dt, Q)
                Q, R = qr(M)
                lys.append(log(abs(diag(R))))

            if has_mpo:
                vL = self.mps.new_vL
                
                if order=='high':
                    self.mps = self.invfree4(self.mps, dt, H)
                elif order=='low':
                    self.mps = self.invfree(self.mps, dt, H)

                self.mps.old_vL = vL
                self.vL = vL
            else:
                vL = self.mps.new_vL

                old_mps = self.mps.copy()
                self.mps = self.rk4(self.mps, dt, H).right_canonicalise()
                self.mps.match_gauge_to(old_mps)

                self.mps.old_vL = vL
                self.vL = vL

        if hasattr(self, 'lys'):
            self.lys = ct([self.lys, array(lys)])
        else:
            self.lys = array(lys)

        if just_max:
            self.q = q
            k = 200
            self.exps = (1/dt)*cs(self.lys, axis=0)[k:]/ar(1, len(self.lys)+1-k)
        else:
            self.Q = Q
            self.exps = cs(self.lys, axis=0)/ed(ar(1, len(self.lys)+1), 1)

        return self.exps, self.lys

    def ed_OTOC(self, T, ops):
        """ed_OTOC: -<[W(t), V(0)], [W(t), V(0)]>

        :param T: times
        :param ops: (V, W)
        """
        assert self.fullH == True
        V, W = ops
        psi_0 = self.mps.recombine().reshape(-1)
        H = self.H
        dt = T[1]-T[0]
        Ws = []
        for t in tqdm(T):
            U = expm(-1j*H*t)
            Wt = cT(U)@W@U
            Ws.append(-re(c(psi_0)@comm(Wt, V)@comm(Wt, V)@psi_0))
        return Ws

    def mps_OTOC(self, T, ops, dt=1e-2):
        """mps_OTOCs: -<[W(t), V(0)], [W(t), V(0)]>

        :param T: times
        :param ops: operators (V, W)
        """
        assert self.W is not None
        # want to use inverse free method since it's time symmetric
        V, W = ops
        mps_0, mpo = self.mps, self.W
        Ws = []
        psi_1 = mps_0 #|s>
        psi_2 = mps_0.copy().apply(V).left_canonicalise() # V|s>
        n2_0 = psi_2.norm_
        T = ct([[0], T])
        for t1, t2 in zip(T, T[1:]):
            for _ in linspace(t1, t2, int((t2-t1)/dt)):
                psi_1 = self.invfree(psi_1, dt)
                psi_2 = self.invfree(psi_2, dt)
            psi_1_ = psi_1.copy().apply(W).left_canonicalise() #WU|s>
            psi_2_ = psi_2.copy().apply(W).left_canonicalise() #WUV|s>
            n1_1 = psi_1_.norm_
            n2_1 = psi_2_.norm_
            for _ in linspace(t1, t2, int((t2-t1)/dt)):
                psi_1_ = self.invfree(psi_1_, -dt)
                psi_2_ = self.invfree(psi_2_, -dt)
            psi_1_.apply(V).left_canonicalise()
            n1_0 = psi_1_.norm_
            psi_1[0] *= n1_0*n1_1
            psi_2[0] *= n2_0*n2_1
            Ws.append(psi_1_.overlap(psi_1_)+psi_2_.overlap(psi_2_)-
                      0.5*re(psi_1_.overlap(psi_2_)))

        return Ws

    def mps_list(self):
        assert self.mps_history
        L, d, D = self.mps.L, self.mps.d, self.mps.D
        return list(map(lambda x: fMPS().deserialize(x, L, d, D, real=True), self.mps_history))

    def mps_evs(self, ops, site):
        assert self.mps_history
        L, d, D = self.mps.L, self.mps.d, self.mps.D
        return array([mps.Es(ops, site)
                    for mps in map(lambda x: fMPS().deserialize(x, L, d, D, real=True), self.mps_history)])

    def mps_energies(self):
        assert self.H is not None
        assert self.mps_history
        return [mps.energy(self.H) for mps in self.mps_list()]

    def ed_energies(self):
        assert self.H is not None
        assert hasattr(self, 'ed_history')
        H = self.H
        H = sum([n_body(a, i, len(H), d=2)
                 for i, a in enumerate(H)], axis=0) if not self.fullH else H
        return self.ed_evs([H])

    def deserialize(self):
        L, d, D = self.mps.L, self.mps.d, self.mps.D
        return list(map(lambda x: fMPS().deserialize(x, L, d, D, real=True), self.mps_history))

    def schmidts(self):
        if hasattr(self, 'ed_history'):
            sch = []
            mpss = []
            for x in tqdm(self.ed_history):
                mps = fMPS().left_from_state(x.reshape([self.mps.d]*self.mps.L))
                sch.append([diag(x) for x in mps.left_canonicalise().Ls])
            return sch[1:]
        else:
            assert self.mps_history
            L, d, D = self.mps.L, self.mps.d, self.mps.D
            sch = []
            for mps in map(lambda x: fMPS().deserialize(x, L, d, D, real=True), self.mps_history):
                sch.append([diag(x) for x in mps.left_canonicalise().Ls])
            return sch[1:]

    def ed_evs(self, ops):
        assert hasattr(self, "ed_history")
        return array([[re(psi.conj()@op@psi) for op in ops] for psi in self.ed_history])

    def clear(self):
        self.mps_history = []
        self.mps = self.mps_0.copy()

    def save(self, loc='data/', exps=True, clear=True):
        assert self.mps_history
        assert hasattr(self, 'exps') if exps else True
        self.id = self.run_name+'_L{}_D{}_N{}'.format(self.mps.L, self.mps.D, len(self.mps_history))
        save(loc+self.id, self.mps_history if not exps else self.lys)
        if exps:
            if hasattr(self, 'q'):
                save(loc+'bases/'+self.id+'_basis', self.q)
                self.mps.store('loc'+'bases/'+self.id+'_state')
            elif hasattr(self, 'Q'):
                save(loc+'bases/'+self.id+'_basis', self.Q)
                self.mps.store(loc+'bases/'+self.id+'_state')
        if clear:
            self.clear()

    def stop(self, loc='data'):
        """stop: save a folder with all the data under loc
        """
        assert not self.fullH
        assert hasattr(self, 'W')
        # make directories - doesn't work if more runs than 10
        # shouldn't be doin more runs than 10
        # careful that there are no files with run name in the folder
        run_dir = os.path.join(loc, self.run_name)
        if not run_dir in map(lambda x: x[:-2], glob.glob(run_dir+'_*')):
            run_dir = run_dir+'_1'
            os.makedirs(run_dir)
        else:
            n = max(list(map(lambda x: int(x[-1]), glob.glob(run_dir+'_*'))))
            run_dir = run_dir+'_'+str(n+1)
            os.makedirs(run_dir)

        self.run_dir = run_dir
        name = 'L{}_D{}_N{}'.format(self.mps.L, self.mps.D, len(self.mps_history))

        self.mps.store(os.path.join(run_dir, name+'_state'))
        self.mps_0.store(os.path.join(run_dir, name+'_initial_state'))

        with open(os.path.join(run_dir, name+'_ham'), 'wb') as f:
            pickle.dump(self.H, f)
        with open(os.path.join(run_dir, name+'_mpo'), 'wb') as f:
            pickle.dump(self.W, f)
        with open(os.path.join(run_dir, name+'_vL'), 'wb') as f:
            pickle.dump(self.vL, f)

        if self.has_run_lyapunov:
            save(os.path.join(run_dir, name+'_inst_exps'), self.lys)
            save(os.path.join(run_dir, name+'_basis'), self.Q)
        return run_dir

    def resume(self, run_name, loc='data', n=None, lys=True):
        run_dir = os.path.join(loc, run_name)
        # resume the most recent run if not specified
        if n is None:
            n = max(list(map(lambda x: int(x[-1]), glob.glob(run_dir+'_*'))))
        self.run_dir = run_dir+'_'+str(n)
        istate, state = sorted(glob.glob(os.path.join(self.run_dir, '*state*')))
        H_loc = glob.glob(os.path.join(self.run_dir, '*ham*'))[0]
        W_loc = glob.glob(os.path.join(self.run_dir, '*mpo*'))[0]
        vL_loc = glob.glob(os.path.join(self.run_dir, '*vL*'))[0]
        self.mps_0, self.mps = fMPS().load(istate), fMPS().load(state)
        with open(H_loc, 'rb') as f:
            self.H = pickle.load(f)
        with open(W_loc, 'rb') as f:
            self.W = pickle.load(f)
        with open(vL_loc, 'rb') as f:
            self.mps.old_vL = pickle.load(f)

        if lys:
            Q_loc = glob.glob(os.path.join(self.run_dir, '*basis*'))[0]
            lys_loc = glob.glob(os.path.join(self.run_dir, '*inst*'))[0]
            self.Q = load(Q_loc)
            self.lys = load(lys_loc)

        return self

    def delete(self):
        print('deleting...',  self.run_dir)
        shutil.rmtree(self.run_dir)

class TestTrajectory(unittest.TestCase):
    """TestF"""
    def setUp(self):
        """setUp"""
        self.tens_0_2 = load('fixtures/mat2x2.npy')
        self.tens_0_3 = load('fixtures/mat3x3.npy')
        self.tens_0_4 = load('fixtures/mat4x4.npy')
        self.tens_0_5 = load('fixtures/mat5x5.npy')

        self.mps_0_1 = fMPS().left_from_state(self.tens_0_2).right_canonicalise(1)

        self.mps_0_2 = fMPS().left_from_state(self.tens_0_2)
        self.psi_0_2 = self.mps_0_2.recombine().reshape(-1)

        self.mps_0_3 = fMPS().left_from_state(self.tens_0_3)
        self.psi_0_3 = self.mps_0_3.recombine().reshape(-1)

        self.mps_0_4 = fMPS().left_from_state(self.tens_0_4)
        self.psi_0_4 = self.mps_0_4.recombine().reshape(-1)

        self.mps_0_5 = fMPS().left_from_state(self.tens_0_5)
        self.psi_0_5 = self.mps_0_5.recombine().reshape(-1)

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
