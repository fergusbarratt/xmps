import os
import glob
import shutil
import unittest
import pickle

from time import time

from .iMPS import iMPS
from .tensor import get_null_space, H as cT, C as c
from .ncon import ncon

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
    def __init__(self, mps_0=None, H=None, run_name=''):
        """__init__

        :param mps_0: initial state
        :param H: hamiltonian
        :param W: mpo (for invfreeint)
        :param T: time steps
        :param run_name: prefix for saving
        """
        self.H = H  # hamiltonian as list of 4x4 mats or big matrix

        self.mps_0 = mps_0.copy() if mps_0 is not None else mps_0
        self.mps = mps_0.copy() if mps_0 is not None else mps_0
        self.Ds = [self.mps.D] if self.mps is not None else []
        self.mps_history = []
        self.run_name = run_name

    def euler(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        return (mps + mps.dA_dt(H)*dt).left_canonicalise()

    def rk4(self, mps, dt, H=None, store=True):
        H = self.H if H is None else H
        if store:
            self.mps_history.append(mps.serialize(real=True))
            self.Ds.append(mps.D)
        k1 = mps.dA_dt(H)
        k2 = (mps+k1/2*dt).dA_dt(H)
        k3 = (mps+k2/2*dt).dA_dt(H)*dt
        k4 = (mps+k3*dt).dA_dt(H)

        return (mps+(k1+2*k2+2*k3+k4)*dt/6).left_canonicalise()

    def eulerint(self, T):
        """eulerint: integrate with euler time steps

        :param T:
        """
        mps, H = self.mps.left_canonicalise(), self.H
        d, D = mps.d, mps.D

        for t in tqdm(T):
            mps = self.euler(mps, T[1]-T[0])

        self.mps = iMPS().deserialize(self.mps_history[-1], d, D, real=True)
        return self

    def rk4int(self, T):
        """rk4int: integrate with rk4 timesteps
        """
        mps, H = self.mps.left_canonicalise(), self.H
        d, D = mps.d, mps.D

        for t in tqdm(T):
            mps = self.rk4(mps, T[1]-T[0])

        self.mps = iMPS().deserialize(self.mps_history[-1], d, D, real=True)
        return self

    def loschmidts(self):
        assert self.mps_history
        d, D = self.mps.d, self.mps.D
        mps_0 = iMPS().deserialize(self.mps_history[0], d, D, real=True)
        les = []
        for mps in (iMPS().deserialize(x, d, D, real=True) for D, x in zip(self.Ds, self.mps_history)):
            les.append(-np.log(np.abs(mps.overlap(mps_0))))
        return np.array(les)
