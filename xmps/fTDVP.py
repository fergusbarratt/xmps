import os
import glob
import shutil
import unittest
import pickle

from time import time

from .fMPS import fMPS
from .tensor import get_null_space, H as cT, C as c
from .ncon import ncon
try:
    from .tdvp.tdvp_fast import tdvp, MPO_TFI
except ModuleNotFoundError:
    print('no finite tdvp module found: module might not work as expected')
    tdvp_available = False
else:
    tdvp_available = True

from .spin import N_body_spins, spins, comm, n_body

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
        self.H = H  # hamiltonian as list of 4x4 mats or big matrix
        self.W = W  # hamiltonian as mpo - required for invfreeint

        self.mps_0 = mps_0.copy() if mps_0 is not None else mps_0
        self.mps = mps_0.copy() if mps_0 is not None else mps_0
        self.Ds = [self.mps.D] if self.mps is not None else []
        self.fullH = fullH
        self.mps_history = []
        self.run_name = run_name
        self.continuous = continuous

    def euler(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        return (mps + mps.dA_dt(H, fullH=self.fullH)*dt).left_canonicalise(mps.D)

    def rk4(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        k1 = mps.dA_dt(H, fullH=self.fullH)*dt
        k2 = (mps+k1/2).dA_dt(H, fullH=self.fullH)*dt
        k3 = (mps+k2/2).dA_dt(H, fullH=self.fullH)*dt
        k4 = (mps+k3).dA_dt(H, fullH=self.fullH)*dt

        return (mps+(k1+2*k2+2*k3+k4)/6).left_canonicalise()

    def invfree(self, mps, dt, W=None, store=True):
        W = W if W is not None else self.W
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        if self.continuous:
            A_old = mps.copy()
        A = tdvp(mps.data, W, 1j*dt/2)
        return fMPS(A[0]) if not self.continuous else fMPS(A[0]).match_gauge_to(A_old)

    def invfree4(self, mps, dt, W=None, store=True):
        """invfree4 fourth order symmetric composition
        """
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        if self.continuous:
            A_old = mps.copy()
        a1, a2, a3 = b5, b4, b3 = (146+5*sqrt(19)) / \
            540, (-2+10*sqrt(19))/135, 1/5
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

        :param T: list of time steps
        :param D: bond dimension to run simulation at
        """
        assert self.W is not None
        mps, H = self.mps, self.H
        L, d, D = mps.L, mps.d, mps.D
        # if D> mps.D:
        #    mps.expand(D)
        # else:
        #    mps.left_canonicalise(D)

        for t in tqdm(T):
            if order == 'high':
                mps = self.invfree4(mps, T[1]-T[0])
            elif order == 'low':
                mps = self.invfree(mps, T[1]-T[0])

        self.mps = fMPS().deserialize(self.mps_history[-1], L, d, D, real=True)
        return self

    def expand(self, D):
        self.mps.expand(D)
        self.Ds[-1] = D
        return self

    def dynamical_expand(self, D, dt, H=None):
        assert self.H is not None or H is not None
        H = self.H if H is None and self.H is not None else H

        self.mps.dynamical_expand(self.H, dt, D, None)
        self.Ds[-1] = D
        return self

    def contract(self, D):
        self.mps.left_canonicalise(D)
        self.Ds[-1] = D
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
        m = 0

        def f(t, v):
            """f_odeint: f acting on real vector

            :param v: Vector: [reals, imags]
            :param t: Time
            """
            bar.update()
            nonlocal m
            m += 1
            if m == 4:
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
            psi_n = expm_multiply(-1j * H*dt, psi_n)
            self.ed_history.append(psi_n)

        self.ed_history = array(self.ed_history)
        self.psi = self.ed_history[-1]
        self.mps = fMPS().left_from_state(
            self.psi.reshape([self.mps.d]*self.mps.L))
        return self

    def lyapunov2(self, T, D=None, t_burn=0):
        self.has_run_lyapunov = True
        H = self.H
        has_mpo = self.W is not None
        if D is not None and t_burn != 0 and not hasattr(self, 'Q'):
            # if MPO supplied - just expand, canonicalise and use inverse free integrator
            # otherwise use dynamical expand: less numerically stable
            # if we already have a basis set - we must be resuming a run
            if has_mpo:
                self.mps = self.mps.right_canonicalise().expand(D)
                self.invfreeint(
                    linspace(0, t_burn, int(50*t_burn)), order='high')
                self.burn_len = int(200*t_burn)
                self.mps_history = []
            else:
                self.mps = self.mps.grow(self.H, 0.1, D).right_canonicalise()
                self.rk4int(linspace(0, 1, 100))

        dt = T[1]-T[0]
        _, J2, _ = self.mps.jac(H, real_matrix=False)
        Js = zl(kron(Sz, J2))*1j
        for t in tqdm(range(len(T))):
            J1, J2, Γ2 = self.mps.jac(H, real_matrix=False)
            J2 = J2+Γ2
            Js += kron(Sz, re(J2))+kron(Sz, im(J2))
            if has_mpo:
                vL = self.mps.new_vL

                self.mps = self.invfree4(self.mps, dt, H)

                self.mps.old_vL = vL
                self.vL = vL

        return eigvalsh(Js)

    def lyapunov(self, T, D=None, thresh=1e-5, conv_window=100,
                 just_max=False,
                 t_burn=2,
                 initial_basis='F2',
                 order='low',
                 k=0):
        self.has_run_lyapunov = True
        H = self.H
        has_mpo = self.W is not None
        if D is not None and t_burn != 0 and not hasattr(self, 'Q'):
            # if MPO supplied - just expand, canonicalise and use inverse free integrator
            # otherwise use dynamical expand: less numerically stable
            # if we already have a basis set - we must be resuming a run
            print('starting pre evolution ... ', end='', flush=True)

            if has_mpo:
                self.mps = self.mps.left_canonicalise().expand(D)
                self.invfreeint(
                    linspace(0, t_burn, int(50*t_burn)), order=order)
                self.burn_len = int(200*t_burn)
                self.mps = self.mps.left_canonicalise()
                self.mps_history = []
            else:
                self.mps = self.mps.grow(self.H, 0.1, D).right_canonicalise()
                self.rk4int(linspace(0, 1, 100))

            print('finished pre evolution')

        if hasattr(self, 'Q'):
            Q = self.Q
        elif initial_basis == 'F2':
            Q = self.mps.tangent_space_basis(H=H, type=initial_basis)
        elif initial_basis == 'eye' or initial_basis == 'rand':
            Q = kron(eye(2), self.mps.tangent_space_basis(
                H=H, type=initial_basis))
        else:
            Q = initial_basis
        Q_ = copy(Q)

        if just_max:
            # just evolve top vector, dont bother with QR
            q = Q[0]
        dt = T[1]-T[0]

        lys = []
        exps = [array([0])]

        lys_ = []
        exps_ = [array([0])]

        self.vs = []

        conv = []
        conv_ = []

        paired = []
        paired_ = []
        for n in tqdm(range(len(T))):
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
                exps.append((exps[-1]*n+lys[-1])/(n+1))
                conv.append(np.mean(np.var(np.array(exps[-conv_window-1:])/dt, axis=0)))
                paired.append(norm(exps[-1]+exps[-1][::-1]))

                M_ = expm_multiply(-J.T*dt, Q_)
                Q_, R_ = qr(M_)
                lys_.append((lys[-1]+log(abs(diag(R_))))/2)
                exps_.append((exps_[-1]*n+lys_[-1])/(n+1))
                conv_.append(np.mean(np.var(np.array(exps_[-conv_window-1:])/dt, axis=0)))
                paired_.append(norm(exps_[-1]+exps_[-1][::-1]))

                if thresh is not None and conv[-1] < thresh:
                    break

            if has_mpo:
                vL = self.mps.new_vL

                if order == 'high':
                    self.mps = self.invfree4(self.mps, dt)
                elif order == 'low':
                    self.mps = self.invfree(self.mps, dt)

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
            if hasattr(self, 'lys_'):
                self.lys_ = ct([self.lys_, array(lys_)])
        else:
            self.lys = array(lys)
            self.lys_ = array(lys_)

        if just_max:
            self.q = q
            self.exps = exps[k+1:n+2]/dt
            self.exps_ = exps_[k+1:n+2]/dt
            T = T[k:n+1]
        else:
            self.Q = Q
            self.exps = exps[k+1:n+2]/dt
            self.exps_ = exps_[k+1:n+2]/dt
            T = T[k:n+1]

        if hasattr(self, 'lys_'):
            return T, (self.exps, self.lys, conv, paired), (self.exps_, self.lys_, conv_, paired_)
        else:
            return T, (self.exps, self.lys, conv, paired)

    def ts_int(self, T, D=None,
               order='high'):
        H = self.H
        assert self.W is not None

        dt = T[1]-T[0]
        Q = eye(self.mps.jac(H).shape[0])
        self.Ms = [Q]

        for t in tqdm(range(len(T))):
            J = self.mps.jac(H)
            self.Ms.append(expm_multiply(J*dt, self.Ms[-1]))

            vL = self.mps.new_vL

            if order == 'high':
                self.mps = self.invfree4(self.mps, dt, H)
            elif order == 'low':
                self.mps = self.invfree(self.mps, dt, H)

            self.mps.old_vL = vL

        return self

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
        psi_1 = mps_0  # |s>
        psi_2 = mps_0.copy().apply(V).left_canonicalise()  # V|s>
        n2_0 = psi_2.norm_
        T = ct([[0], T])
        for t1, t2 in zip(T, T[1:]):
            for _ in linspace(t1, t2, int((t2-t1)/dt)):
                psi_1 = self.invfree(psi_1, dt)
                psi_2 = self.invfree(psi_2, dt)
            psi_1_ = psi_1.copy().apply(W).left_canonicalise()  # WU|s>
            psi_2_ = psi_2.copy().apply(W).left_canonicalise()  # WUV|s>
            n1_1 = psi_1_.norm_
            n2_1 = psi_2_.norm_
            for _ in linspace(t1, t2, int((t2-t1)/dt)):
                psi_1_ = self.invfree(psi_1_, -dt)
                psi_2_ = self.invfree(psi_2_, -dt)
            psi_1_.apply(V).left_canonicalise()
            n1_0 = psi_1_.norm_
            psi_1[0] *= n1_0*n1_1
            psi_2[0] *= n2_0*n2_1
            Ws.append(psi_1_.overlap(psi_1_)+psi_2_.overlap(psi_2_) -
                      0.5*re(psi_1_.overlap(psi_2_)))

        return Ws

    def mps_list(self):
        if hasattr(self, 'ed_history') and not self.mps_history:
            for x in tqdm(self.ed_history):
                mps = fMPS().left_from_state(
                    x.reshape([self.mps.d]*self.mps.L)).left_canonicalise()
                self.mps_history.append(mps.serialize(real=True))
        assert self.mps_history
        L, d, D = self.mps.L, self.mps.d, self.mps.D
        return [fMPS().deserialize(x, L, d, D, real=True) for D, x in zip(self.Ds, self.mps_history)]

    def mps_evs(self, ops, site):
        if hasattr(self, 'ed_history') and not self.mps_history:
            for x in self.ed_history:
                mps = fMPS().left_from_state(
                    x.reshape([self.mps.d]*self.mps.L)).left_canonicalise()
                self.mps_history.append(mps.serialize(real=True))
        assert self.mps_history
        L, d, D = self.mps.L, self.mps.d, self.mps.D
        return array([mps.Es(ops, site)
                      for mps in (fMPS().deserialize(x, L, d, D, real=True) for D, x in zip(self.Ds, self.mps_history))])

    def mps_energies(self):
        assert self.H is not None
        assert self.mps_history
        return np.array([mps.left_canonicalise().energy(self.H) for mps in self.mps_list()])

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
            print('calculating mps from ed...')
            for x in tqdm(self.ed_history):
                mps = fMPS().left_from_state(
                    x.reshape([self.mps.d]*self.mps.L))
                sch.append([diag(x) for x in mps.left_canonicalise().Ls])
            return sch
        else:
            assert self.mps_history
            L, d, D = self.mps.L, self.mps.d, self.mps.D
            sch = []
            for mps in (fMPS().deserialize(x, L, d, D, real=True) for D, x in zip(self.Ds, self.mps_history)):
                sch.append([diag(x) for x in mps.left_canonicalise().Ls])
            return sch

    def loschmidt(self):
        if hasattr(self, 'ed_history'):
            sch = []
            mpss = []
            print('calculating mps from ed...')
            for x in tqdm(self.ed_history):
                mps = fMPS().left_from_state(x.reshape([self.mps.d]*self.mps.L))
                sch.append([diag(x) for x in mps.left_canonicalise().Ls])
            return sch
        else:
            assert self.mps_history
            L, d, D = self.mps.L, self.mps.d, self.mps.D
            psi_0 = fMPS().deserialize(self.mps_history[0], L, d, D, real=True)
            lss = []
            for psi in map(lambda x: fMPS().deserialize(x, L, d, D, real=True), self.mps_history):
                lss.append(np.abs(psi_0.overlap(psi)))
            return lss

    def von_neumann(self, i=None):
        sch = array(self.schmidts())
        S_max = array([max([-re(s@log(s)) for s in S]) for S in sch])
        return S_max

    def renyi(self, α=2):
        sch = array(self.schmidts())
        R = array([max([1/(1-α)*log(sum(s**α)) for s in S]) for S in sch])
        return R

    def ed_evs(self, ops):
        assert hasattr(self, "ed_history")
        return array([[re(psi.conj()@op@psi) for op in ops] for psi in self.ed_history])

    def clear(self):
        self.mps_history = []
        self.mps = self.mps_0.copy()
        if hasattr(self, 'lys'):
            delattr(self, 'lys')

    def save(self, loc='data/', exps=True, clear=True):
        assert self.mps_history
        assert hasattr(self, 'exps') if exps else True
        self.id = self.run_name + \
            '_L{}_D{}_N{}'.format(self.mps.L, self.mps.D,
                                  len(self.mps_history))
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
        name = 'L{}_D{}_N{}'.format(
            self.mps.L, self.mps.D, len(self.mps_history))

        self.mps.store(os.path.join(run_dir, name+'_state'))
        self.mps_0.store(os.path.join(run_dir, name+'_initial_state'))

        with open(os.path.join(run_dir, name+'_ham'), 'wb') as f:
            pickle.dump(self.H, f)
        with open(os.path.join(run_dir, name+'_mpo'), 'wb') as f:
            pickle.dump(self.W, f)
        if hasattr(self, 'vL'):
            with open(os.path.join(run_dir, name+'_vL'), 'wb') as f:
                pickle.dump(self.vL, f)

        if self.has_run_lyapunov:
            save(os.path.join(run_dir, name+'_inst_exps'), self.lys)
            if hasattr(self, 'lys_'):
                save(os.path.join(run_dir, name+'_inst_exps_'), self.lys_)
            if hasattr(self, 'Q'):
                save(os.path.join(run_dir, name+'_basis'), self.Q)
        return run_dir

    def resume(self, run_name, loc='data', n=None, lys=True):
        run_dir = os.path.join(loc, run_name)
        # resume the most recent run if not specified
        if n is None:
            n = max(list(map(lambda x: int(x[-1]), glob.glob(run_dir+'_*'))))
        self.run_dir = run_dir+'_'+str(n)
        istate, state = sorted(
            glob.glob(os.path.join(self.run_dir, '*state*')))

    def copy(self):
        return copy(self)
