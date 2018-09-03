import unittest
from time import time
from fMPS import fMPS
from ncon import ncon
from spin import N_body_spins, spins
from numpy import array, linspace, real as re, reshape, sum, swapaxes as sw
from numpy import tensordot as td, squeeze, trace as tr, expand_dims as ed
from numpy import load, isclose, allclose, zeros_like as zl, prod, imag as im
from numpy import log, abs, diag, cumsum as cs, arange as ar, eye, kron as kr
from numpy.random import randn
from scipy.linalg import sqrtm, expm, norm, null_space as null, cholesky as ch
from scipy.integrate import odeint
from numpy import split, concatenate as ct
from numpy.linalg import inv, qr
from tensor import get_null_space, H as cT, C as c
from matplotlib import pyplot as plt
from functools import reduce
Sx, Sy, Sz = spins(0.5)
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

def trajectory(mps_0, H, dt, N, D=None, m=None, plot=True, timeit=False):
    """trajectory: calculate trajectory and optionally lyapunov exponents.
                   Now with rk4!

    :param mps_0: initial mps
    :param H: hamiltonian
    :param dt: timestep
    :param N: no. of time steps
    :param m: calculate exponents every m steps - don't calculate them if m is None
    """
    def rk4(mps, H, dt):
        k1 = mps.dA_dt(H)*dt
        k2 = (mps+k1/2).dA_dt(H)*dt
        k3 = (mps+k2/2).dA_dt(H)*dt
        k4 = (mps+k3).dA_dt(H)*dt

        return mps+(k1+2*k2+2*k3+k4)/6

    d, L, D = mps_0.d, mps_0.L, mps_0.D

    mps_n = mps_0
    psi_n = mps_0.recombine().reshape(-1)

    Ly = []
    Q = eye(D*4)

    W_ed = [psi_n]
    W_mps = [mps_n]
    t1 = time()
    for n in range(N):
        if d == 2 and m is not None and not n%m:
            A = mps_n.left_canonicalise().data
            pr = mps_n.left_null_projector

            # for full size, D=d=L=2, left canonical mps this is the only diagram
            J = -1j*ncon([A[0].conj(), pr(1), H.reshape([d, d]*L)]+[A[0]], 
                         [[1, 5, 6], [-1, -2, 2, 6], [1, 2, 3, -3], [3, 5, -4]])
            J = J.reshape(J.shape[0]*J.shape[1], -1)
            J = kr(eye(2), re(J))+kr(-2j*Sy, im(J))
            Q = expm(dt*J)@Q
            Q, R = qr(Q)

            Ly.append(log(abs(diag(R))))

        mps_n = rk4(mps_n, H, dt)
        psi_n = expm(-1j*H*dt)@psi_n

        W_ed.append(psi_n)
        W_mps.append(mps_n)

    t2 = time()
    if timeit:
        print(t2-t1)
    if plot:
        T1 = linspace(0, N*dt, len(Ly))
        T2 = linspace(0, N*dt, len(W_mps))
        if m is not None:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(T1, (1/(m*dt))*cs(Ly, axis=0)/ed(ar(1, len(Ly)+1), 1))
            _, Sy1, _ = N_body_spins(0.5, 1, L)
            ax[1].plot(T2, [m.E(Sy, 0) for m in W_mps])
            ax[1].plot(T2, [c(p)@Sy1@p for p in W_ed])
        else:
            _, Sy1, _ = N_body_spins(0.5, 1, L)
            plt.plot(T2, [m.E(Sy, 0) for m in W_mps])
            plt.plot(T2, [re(c(p)@Sy1@p) for p in W_ed])

class Trajectory(object):
    """Trajectory"""

    def __init__(self, mps_0, H):
        """__init__

        :param mps_0: initial state
        :param H: hamiltonian
        :param T: time steps
        """
        self.H = H
        self.mps = mps_0
        self.history = []

    def euler(self, dt):
        self.history.append(mps.serialize())
        mps, H = self.mps, self.H
        self.mps =  mps + mps.dA_dt(H)*dt
        return self

    def rk4(self, dt):
        mps, H = self.mps, self.H
        k1 = mps.dA_dt(H)*dt
        k2 = (mps+k1/2).dA_dt(H)*dt
        k3 = (mps+k2/2).dA_dt(H)*dt
        k4 = (mps+k3).dA_dt(H)*dt
        self.history.append(mps.serialize())

        self.mps = mps+(k1+2*k2+2*k3+k4)/6
        return self

    def odeint(self, T, plot=False, timeit=False):
        """odeint: pass to scipy odeint:

        :param mps_0: initial 
        :param plot: plot the result
        :param timeit: time odeint
        """
        def f_odeint(v, t, L, d, D, H):
            """f_odeint

            :param v: Vector: [reals, imags]
            :param t: Time
            :param L: Length
            :param d: Local hilbert space dimension
            :param D: Bond dimension
            :param H: Hamiltonian
            """
            v_re, v_im = split(v, 2)
            v_ = fMPS().deserialize(v_re+1j*v_im, L, d, D).dA_dt(H).serialize()
            return ct([re(v_), im(v_)])

        mps_0, H = self.mps, self.H
        L, d, D = mps_0.L, mps_0.d, mps_0.D
        v = mps_0.serialize()
        t1 = time()
        traj = odeint(f_odeint, ct([re(v), im(v)]), T, args=(L, d, D, H))
        t2 = time()
        if timeit:
            print(t2-t1)
        traj = reduce(lambda r, i: r+1j*i, split(traj.T, 2))
        if plot:
            plt.plot(T, [mps.E(Sy, 0) for mps in map(lambda x: fMPS().deserialize(x, L, d, D), traj.T)])

        self.history = traj.T
        self.mps = fMPS().deserialize(traj.T[-1, :], L, d, D)
        return self

class TestTrajectory(unittest.TestCase):
    """TestF"""
    def setUp(self):
        """setUp"""
        self.tens_0_2 = load('mat2x2.npy')
        self.tens_0_3 = load('mat3x3.npy')
        self.tens_0_4 = load('mat4x4.npy')

        self.mps_0_1 = fMPS().left_from_state(self.tens_0_2).right_canonicalise(1)

        self.mps_0_2 = fMPS().left_from_state(self.tens_0_2)
        self.psi_0_2 = self.mps_0_2.recombine().reshape(-1)

        self.mps_0_3 = fMPS().left_from_state(self.tens_0_3)
        self.psi_0_3 = self.mps_0_3.recombine().reshape(-1)

        self.mps_0_4 = fMPS().left_from_state(self.tens_0_4)
        self.psi_0_4 = self.mps_0_4.recombine().reshape(-1)

    def test_exps_2(self):
        """test_exps_2: 2 spins: 100 timesteps, expectation values within tol=1e-2 of ed results"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        H = Sz1@Sz2 + Sx1+Sx2
        mps_0 = self.mps_0_2
        psi_0 = self.psi_0_2
        dt = 1e-2
        N = 100 
        tol = 1e-2
        mps_n = mps_0
        psi_n = psi_0
        for _ in range(N):
            self.assertTrue(isclose(psi_n.conj()@Sx1@psi_n, mps_n.E(Sx, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy1@psi_n, mps_n.E(Sy, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz1@psi_n, mps_n.E(Sz, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sx2@psi_n, mps_n.E(Sx, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy2@psi_n, mps_n.E(Sy, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz2@psi_n, mps_n.E(Sz, 1), atol=tol))

            mps_n = mps_n + mps_n.dA_dt(H)*dt
            psi_n = expm(-1j*H*dt)@psi_n

    def test_exps_3(self):
        """test_exps_3: 3 spins: 100 timesteps, expectation values within tol=1e-2 of ed results"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 3)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 3)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 3)
        H = Sz1@Sz2 + Sz2@Sz3 + Sx1+Sx2+Sx3
        mps_0 = self.mps_0_3
        psi_0 = self.psi_0_3
        dt = 1e-2
        N = 100 
        tol = 1e-2
        mps_n = mps_0
        psi_n = psi_0
        for _ in range(N):
            self.assertTrue(isclose(psi_n.conj()@Sx1@psi_n, mps_n.E(Sx, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy1@psi_n, mps_n.E(Sy, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz1@psi_n, mps_n.E(Sz, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sx2@psi_n, mps_n.E(Sx, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy2@psi_n, mps_n.E(Sy, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz2@psi_n, mps_n.E(Sz, 1), atol=tol))

            mps_n = mps_n + mps_n.dA_dt(H)*dt
            psi_n = expm(-1j*H*dt)@psi_n

    def test_exps_4(self):
        """test_exps_4: 4 spins: 100 timesteps, expectation values within tol=1e-2 of ed results"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)
        H = Sz1@Sz2 +Sz2@Sz3 + Sz3@Sz4 + Sx1+Sx2+Sx3+Sx4
        mps_0 = self.mps_0_4
        psi_0 = self.psi_0_4
        dt = 1e-2
        N = 100 
        tol = 1e-2
        mps_n = mps_0
        psi_n = psi_0
        for _ in range(N):
            self.assertTrue(isclose(psi_n.conj()@Sx1@psi_n, mps_n.E(Sx, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy1@psi_n, mps_n.E(Sy, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz1@psi_n, mps_n.E(Sz, 0), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sx2@psi_n, mps_n.E(Sx, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sy2@psi_n, mps_n.E(Sy, 1), atol=tol))
            self.assertTrue(isclose(psi_n.conj()@Sz2@psi_n, mps_n.E(Sz, 1), atol=tol))

            mps_n = mps_n + mps_n.dA_dt(H)*dt
            psi_n = expm(-1j*H*dt)@psi_n

    def test_trajectory_2_D_1(self):
        """test_trajectory_2: 2 spins, lyapunov trajectory, initial state has bond dimension 1"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps_0 = self.mps_0_1
        H = 4*(Sz1@Sz2 +Sy1@Sy2+Sx1@Sx2) + (Sx1-Sz2)
        N, dt = 50, 1e-2
        print('\nt1: ', end='')
        trajectory(mps_0, H, dt, N, plot=True, timeit=True)
        plt.show()
        print('t2: ', end='')
        Trajectory(mps_0, H).odeint(linspace(0, N*dt, N), plot=False, timeit=True)

    def test_trajectory_2_no_truncate(self):
        """test_trajectory_2: 2 spins, lyapunov trajectory"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)
        mps_0 = self.mps_0_2
        H = Sz1@Sz2 + Sx1+Sx2
        N = 100
        dt = 1e-1
        print('\nt1: ', end='')
        trajectory(mps_0, H, dt, N, plot=False, timeit=True)
        print('t2: ', end='')
        Trajectory(mps_0, H).odeint(linspace(0, N*dt, N), plot=False, timeit=True)
        # odeint up to 3x faster!

    def test_trajectory_3_no_truncate(self):
        """test_trajectory_3: 3 spins, lyapunov trajectory"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 3)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 3)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 3)
        mps_0 = self.mps_0_3
        H = Sz1@Sz2+Sz2@Sz3 + Sx1+Sx2+Sx3
        dt, N = 1e-1, 100
        print('\nt1: ', end='')
        trajectory(mps_0, H, dt, N, plot=False, timeit=True)
        print('t2: ', end='')
        Trajectory(mps_0, H).odeint(linspace(0, N*dt, N), plot=False, timeit=True)

    def test_trajectory_4_no_truncate(self):
        """test_trajectory_4: 4 spins, lyapunov trajectory"""
        Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 4)
        Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 4)
        Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 4)
        Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 4)
        mps_0 = self.mps_0_4
        H = Sz1@Sz2+Sz2@Sz3+Sz3@Sz4 + Sx1+Sx2+Sx3+Sx4
        dt, N = 1e-1, 100
        print('\nt1: ', end='')
        trajectory(mps_0, H, dt, N, plot=False, timeit=True)
        print('t2: ', end='')
        Trajectory(mps_0, H).odeint(linspace(0, N*dt, N), plot=False, timeit=True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
