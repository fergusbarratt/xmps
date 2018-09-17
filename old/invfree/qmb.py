'''n_body versions of local operators'''
import unittest
from qutip import sigmam, sigmap, sigmax, sigmay, sigmaz, tensor, qeye
from numpy import swapaxes, array, identity, allclose, sqrt, zeros, reshape
from numpy import tensordot
from itertools import product
from numpy.random import randn, rand
from spin import *
        

def S_n_body(op, i, n, tensorise=True):
    """S_n_body: n_body versions of local operators replace
    tensor with np.array for array of objects

    :param op: operator to tensor into chain of identities
    :param i: site for operator
    :param n: length of chain
    :param tensorise: if True return a tensor, if False return an array of matrices
    """
    if not i < n:
        raise Exception("i must be less than n")
    l = [op**m for m in map(lambda j: int(not i-j), range(n))]
    if tensorise:
        return tensor(l)
    else:
        return array([d.full() for d in l])


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


def anisotropic_heisenberg_mpo(h, J, Jz, N):
    """anisotropic_heisenberg_mpo: an_heis as a matrix product operator
    """
    first = array([[-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
    bulk = array([[I, O, O, O, O],
                  [sp, O, O, O, O],
                  [sm, O, O, O, O],
                  [sz, O, O, O, O],
                  [-h*sz, J/2*sm, J/2*sp, Jz*sz, I]])
    last = array([[I], [sp], [sm], [sz], [-h*sz]])
    mpo = [first]+[bulk]*(N-2)+[last]
    for i, mats in enumerate(mpo):
        mpo[i] = swapaxes(swapaxes(mats, 0, 2), 1, 3)
    return mpo

def anisotropic_heisenberg(h, J, Jz, N):
    """anisotropic_heisenberg: full, 2**N an_heis hamiltonian
    """
    NN = sum([J/2*(Sp(i, N)*Sm(i+1, N) +
                   Sm(i, N)*Sp(i+1, N)) +
              Jz*Sz(i, N)*Sz(i+1, N)
              for i in range(N-1)])
    S = -sum([h*Sz(i, N) for i in range(N)])
    return NN + S

def local_anisotropic_heisenberg(h, J, Jz):
    """local_anisotropic_heisenberg, nearest neighbour bit
    """
    h_i = J/2*(Sp(0, 2)*Sm(1, 2) + Sm(0, 2)*Sp(1, 2)) + \
              Jz*Sz(0, 2)*Sz(1, 2) - h*(Sz(0, 2))
    return h_i.full()

def local_anisotropic_heisenbergs(h, J, Jz, N):
    """local_anisotropic_heisenbergs, nearest neighbour bits along a chain
    """
    h_bulk = (J/2*(Sp(0, 2)*Sm(1, 2) + Sm(0, 2)*Sp(1, 2)) +
              Jz*Sz(0, 2)*Sz(1, 2) - h*(Sz(0, 2))).full()
    h_N_1 = (J/2*(Sp(0, 2)*Sm(1, 2) + Sm(0, 2)*Sp(1, 2)) +
              Jz*Sz(0, 2)*Sz(1, 2) - h*(Sz(0, 2)+Sz(1, 2))).full()
    return [h_bulk]*(N-2) + [h_N_1]

def local_ising(J, g):
    h_i = -(J*Sz(0, 2)*Sz(1, 2) + (g/2)*(Sx(0, 2) + Sx(1, 2)))
    return h_i.full()

def local_isings(J, g, N):
    h_bulk =  -(J*Sz(0, 2)*Sz(1, 2) + g*(Sx(0, 2))).full()
    h_N_1 = -(J*Sz(0, 2)*Sz(1, 2) + g*(Sx(0, 2) + Sx(1, 2))).full()
    return  [h_bulk]*(N-2) + [h_N_1]

def local_XX(J, g):
    h_i = -J*(Sz(0, 2) +
             (1/2)*Sx(0, 2)*Sx(1, 2) + (1/2)*Sy(0, 2)*Sy(1, 2))
    return h_i.full()

class TestHamiltonians(unittest.TestCase):

    def setUp(self):
        J = rand()
        Jz = rand()
        h = rand()
        L = 10

        self.H1 = local_anisotropic_heisenbergs(J, Jz, h, L)
        self.H2 = anisotropic_heisenberg(J, Jz, h, L)

    def test_local_anisotropic_heisenbergs(self):
        H1, H2 = self.H1, self.H2
        b = []
        for i, h in enumerate(self.H1):
            # create list of full size tensors by adding identities on left and right
            b.append(tensor([qeye(2)]*i + [h] + [qeye(2)]*(len(H1)-i-1)))
        H1_ = sum(b)
        self.assertTrue(allclose(self.H2.full(), sum(b).full()))

if __name__ == '__main__':
    unittest.main()
