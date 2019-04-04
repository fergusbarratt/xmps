import unittest
from numpy.random import randint, rand, randn
from numpy import zeros, identity, array, transpose, isclose, reshape
from numpy import dot, tensordot, concatenate, sum, trace as tr, real, imag
from numpy import stack, swapaxes, allclose, isclose, abs, squeeze, kron
from numpy import ravel, real_if_close, sqrt
from numpy.linalg import cholesky, inv, eigvals, norm
from iMPS import iMPS
from fMPS import fMPS
from qmb import local_XX, local_anisotropic_heisenberg, sigmaz, sigmax, sigmay
from qmb import local_anisotropic_heisenbergs, sigmaz, anisotropic_heisenberg
from tensor import r_eigenmatrix, l_eigenmatrix, H, T, C, get_null_space
from tensor import p, basis_iterator, structure, single_basis_iterator
from tensor import detensorise
from mps_examples import comp_z
from spin import spinHamiltonians
from scipy.optimize import root
from itertools import product
import matplotlib.pyplot as plt
from time import time


def iA_dot(mps, h_, K0, ops=[], l0=None, r0=None, A_gauge='m', K_method='krylov', testing=False, profile=True):
    """iA_dot: construct and return iA_dot

    :param mps: initial state
    :param h_: nearest neighbour hamiltonian: mps.d**2 x mps.d**2
    :param K0: initial guess for K iteration. 
               about two times faster to use previous guess
    :param l0, r0: initial guesses for arnoldi iteration for canonicalisation
    :param ops: operators to calculate EVs of 
    :param A_gauge: gauge of A matrix
    :param K_method: iterative method to use for determining K
    :param testing: returns intermediate products, calculates several tests
    :param profile: prints times taken for different operations
    """
    # Declare variables - prepare will set them all
    d=D=eta=l=l_half=inv_l_half=r=r_half=inv_r_half=vL=vL_=L_=None
    # Energy - set nonlocally
    energy=None

    def prepare(mps, h_, ops):
        """prepare: spits out global variables generated from the matrix product state

        :param mps: matrix product state
        :param h_: hamiltonian: d**2 x d**2
        :param ops: list of operators to calculate expectation values of
        """

        # basic
        nonlocal d
        nonlocal D

        # to do with canonicalisations
        nonlocal eta
        nonlocal l
        nonlocal l_half
        nonlocal inv_l_half
        nonlocal r
        nonlocal r_half
        nonlocal inv_r_half
        nonlocal vL

        # testing intermediates
        nonlocal L_
        nonlocal vL_

        mps.canonicalise(A_gauge)
        evs = array([[mps.E(op, A_gauge) for op in ops]])

        d = mps.d
        D = mps.D
        h = detensorise(h_, d)  # h[u, v, s, t] = <u, v|h_|s, t>

        A = mps.data[0]

        # find largest eigenmatrices
        eta, l, r = mps.eigs() 

        # Square roots and pre-compute inverses
        r_half = cholesky(r)
        l_half = cholesky(l) 

        inv_l_half = inv(l_half)
        inv_r_half = inv(r_half)

        # barred are matrices, unbarred tensors
        L = H(A) @ l_half 
        L_ = reshape(swapaxes(L, 0, 1), [mps.D, mps.d*mps.D]) 

        vL_ = get_null_space(L_)
        vL = reshape(vL_, [mps.d, mps.D, -1])

        return A, h, evs

    def B(x):
        """B: parametrisation of B matrices with gauge fixed

        :param x: (D*(d-1), D) matrix of free elements of B
        """

        return inv_l_half @ vL @ x @ inv_r_half

    def lHAA(A, h):
        """lHAA: (l|H_{AA}^{AA}, p3 Haegeman

        :param A: Matrix of umps
        :param h: nearest neighbour hamiltonian, in tensor product form 
        i.e. 4x4 for (2x2)\cdot(2x2)
        """
        lH = tensordot(h, 
                       dot(H(swapaxes(dot(A, A), 1, 2)), 
                           l @ swapaxes(dot(A, A), 1, 2)), 
                    [[0, 1, 2, 3], [0, 1, 3, 4]])
                        
        return lH

    def E_1(A, h, K0):
        """E_1: iteratively find the pseudo inverse of (1-E). p3 Haegeman

        :param A: Matrix of umps
        :param h: nearest neighbour hamiltonian, in tensor product form 
        i.e. 4x4 for (2x2)\cdot(2x2)
        """
        lH = lHAA(A, h)

        nonlocal energy
        energy = real_if_close(tr(lH@r))

        hl = energy*l 

        def O(K):
            """O: function of which K should be the root
            scipy root doesn't work with complex numbers or ndim matrices (maybe)
            takes a list of length 2*mps.D**2, with first half real parts, second
            half imaginary parts. 
            returns list in the same format

            :param K: guess for K. for correct K will return zeros
            """

            K = reshape(K[:D**2]+1j*K[D**2:], (D, D))

            AKA = sum(H(A) @ K @ A, axis=0)
            LHS = K - AKA + tr(K@r)*l 
            RHS = lH - hl 
            O_n = reshape(LHS - RHS , (D**2,))
            O_n = concatenate([real(O_n), imag(O_n)])
            return O_n

        K0 = reshape(K0, (D**2,))
        K0 = concatenate([real(K0), imag(K0)])

        K = root(O, K0, method=K_method).x
        K = reshape(K[:D**2]+1j*K[D**2:], (D, D))
        return K

    def F(A, K, h):
        """F: get F tensor (D(d-1)xD) from p3 c2 Haegeman
        Makes heavy use of array broadcasting rules

        :param h: nearest neighbour hamiltonian (d**2xd**2)
        """
        C = tensordot(h, dot(A, A), [[2, 3], [0, 2]]) 

        FL = H(vL) @ l_half
        FM = C
        FR = r @ H(A) @ inv_r_half
        F1 = sum(tensordot(FL, C @ FR, [[0, -1], [0, -2]]), axis=1)
        #F1a = sum(sum(FL @ FM @ FR, axis=0), axis=0)
        #print(allclose(F1, F1a))

        FL = H(vL) @ inv_l_half
        FM = sum(H(A) @ l @ C + K @ A, axis=0)
        FR = r_half
        F2 = sum(FL @ FM @ FR, axis=0)
        return F1 + F2

    if profile:
        t0 = time()
        A, h, evs = prepare(mps, h_, ops)

        t1 = time()
        K = E_1(A, h, K0)

        t2 = time()
        x = F(A, K, h)

        t3 = time()
        B = B(x)

        t4 = time()
        q = real_if_close(tr(H(x) @ x))

        print('\n')
        print('-: ', t1-t0, end='\n')
        print('K: ', t2-t1, end='\n')
        print('F: ', t3-t2, end='\n')
        print('B: ', t4-t3, end='\n')

        iA_dot = -1j*B
    else:
        A, h, evs = prepare(mps, h_, ops)
        x = F(A, E_1(A, h, K0), h)
        iA_dot = -1j*B(x)
        q = real_if_close(tr(H(x) @ x))

    if testing:
        #assert isclose(eta, 1)
        assert isclose(tr(l @ r), 1)
        assert allclose(mps.transfer_matrix().aslinearoperator() @ ravel(r), eta*ravel(r))
        assert allclose(mps.transfer_matrix().aslinearoperator().H @ ravel(l), eta*ravel(l))
        assert allclose(r, r.conj().T)
        assert allclose(l, l.conj().T)
        assert all(eigvals(r) > 0)
        assert all(eigvals(l) > 0)

        assert allclose(H(r_half) @ r_half, r)
        assert allclose(H(l_half) @ l_half, l)

        assert allclose(inv_r_half @ H(inv_r_half), inv(r))
        assert allclose(inv_l_half @ H(inv_l_half), inv(l))
        K0 = randn(D, D) + 1j*randn(D, D) 
        assert allclose(tr(E_1(A, h, K0) @ r), 0)

    if testing:
        K0 = randn(D, D) + 1j*randn(D, D) 
        Ks = [E_1(A, h, K0) for _ in range(10)]
        return L_, vL_, l, r, eta, iA_dot, A, Ks, h
    else:
        return iA_dot, evs, energy, sqrt(q), K, l, r

def iTDVP(mps, h, dt, n, ops=[], A_gauge='m', K_method='krylov', noisy=True):
    """iTDVP: time evolve TI iMPS with (nearest neigbour) hamiltonian h
    time step dt for n steps, calculating EVs of ops along the way
    
    return:  imps object, ops expectation list, list of energies
    """
    def euler_step(mps, h, dt, K0, l0, r0):
        A_dot, evs, H, q, K0, l0, r0 = iA_dot(mps, h, K0, ops, l0, r0, A_gauge=A_gauge, K_method=K_method)
        return iMPS([mps.data[0] + dt*A_dot]), evs, H, q, K0, l0, r0
    
    # euler update internal matrix
    ret = [] 
    es = []
    K0 = rand(mps.D, mps.D) + 1j*rand(mps.D, mps.D)
    l0, r0 = None, None
    for i in range(n):
        if noisy:
            print(i, ',', sep='', end='',  flush=True)
        t0 = time()
        mps, evs, H, q, K0, l0, r0 = euler_step(mps, h, dt, K0, l0, r0)
        ret.append(evs)
        es.append(H)
        t1 = time()
        print('T:', t1-t0)
        print('tol: ', q)

    return mps, concatenate(ret, axis=0), es

class TestTDVP(unittest.TestCase):

    def setUp(self):
        N = 1 
        L_min, L_max = 7, 8 
        D_min, D_max = 5, 6 
        d_min, d_max = 2, 3
        p_min, p_max = 1, 2  # always 1 for now 
        self.i_rand_cases = [iMPS().random(randint(d_min, d_max),
                                         randint(D_min, D_max),
                                         period=randint(p_min, p_max))
                           for _ in range(N)]
        self.f_rand_cases = [fMPS().random(randint(L_min, L_max),
                                         randint(d_min, d_max),
                                         randint(D_min, D_max))
                            for _ in range(N)]

    def test_iA_dot(self):
        h = spinHamiltonians(0.5).heisenberg_ferromagnet(1) 
        print(h.shape)
        for case in self.i_rand_cases:
            d, D = case.d, case.D
            K0 = rand(D, D) + 1j*rand(D, D)
            L, vL, l, r, eta, B, A, Ks, h = iA_dot(case, h, K0, testing=True)
            #self.assertTrue(isclose(eta, 1))
            # Lv_L = 0
            self.assertTrue(allclose(L @ vL, 0))
            # v_L.dagger v_l = I
            self.assertTrue(allclose(H(vL) @ vL, identity(D)))
            # gauge condition on B
            self.assertTrue(allclose(tensordot(l, transpose(tensordot(B, C(A), [0, 0]), [0, 2, 1, 3]), [[0, 1], [0, 1]]), 0))
            # Ks: Assert E_1 gives the same answer when applied N times
            K = Ks[0]
            K1 = Ks
            K2 = Ks[1:]
            self.assertTrue(all([allclose(k1, k2) for k1, k2 in zip(K1, K2)]))
            # tr(Kr) = 0
            self.assertTrue(isclose(tr(K @ r), 0))

    def test_iA_dot_detensorise(self):
        initial = iMPS().random(2, 5)
        h_f = [(kron(sigmaz().full(), identity(2)))]
        K0 = rand(5, 5) + 1j*rand(5, 5)
        _, _, _, _, _, _, _, _, h = iA_dot(initial, h_f[0], K0, testing=True)
        self.assertTrue(allclose(h[:, 0, :, 0], identity(2)))
        self.assertTrue(allclose(h[0, :, 0, :], sigmaz().full()))
        h_f = [(kron(sigmaz().full(), sigmaz().full()))]
        _, _, _, _, _, _, _, _, h = iA_dot(initial, h_f[0], K0, testing=True)
        self.assertTrue(allclose(h[:, 0, :, 0], sigmaz().full()))
        self.assertTrue(allclose(h[0, :, 0, :], sigmaz().full()))
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
