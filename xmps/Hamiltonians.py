from functools import reduce
from itertools import product
from .spin import paulis
from numpy import zeros, kron, trace, eye, allclose
import numpy as np

Sx, Sy, Sz = paulis(0.5)
S = {'I': eye(2), 'X': Sx, 'Y': Sy, 'Z': Sz}

class Hamiltonian:
    """Hamiltonian: string of terms in local hamiltonian.
       Just do quadratic spin 1/2
       ex. tfim = Hamiltonian({'ZZ': 1, 'X': λ}) = Hamiltonian({'ZZ': 1, 'IX': λ/2, 'XI': λ/2})
       for parity invariant specify can single site terms ('X')
       otherwise 'IX' 'YI' etc."""

    def __init__(self, strings=None):
        self.strings = strings
        if strings is not None:
            for key, val in {key:val for key, val in self.strings.items()}.items():
                if len(key)==1:
                    self.strings['I'+key] = val/2
                    self.strings[key+'I'] = val/2
                    self.strings.pop(key)

    def __str__(self):
        return str(self.strings)

    def _squeeze(self):
        self.strings = {key:val for key, val in self.strings.items() if not allclose(val, 0)}
        return self

    def to_matrix(self):
        assert self.strings is not None
        h_i = zeros((4, 4))+0j
        for js, J in self.strings.items():
            h_i += J*reduce(kron, [S[j] for j in js])
        self._matrix = h_i
        return h_i

    def to_mpo(self, L):
        def heis_mpo(Jx, Jy, Jz, hx, hy, hz):
            I = np.eye(2)
            Sx = np.array([[0., 1.], [1., 0.]])
            Sy = np.array([[0, -1j], [1j, 0]])
            Sz = np.array([[1., 0.], [0., -1.]])
            d = 2

            chi = 5
            W = np.zeros((chi, chi, d, d))*1j
            W[0, 0] += I
            W[0, 1] += Sz
            W[0, 2] += Sy
            W[0, 3] += Sx
            W[0, 4] += hz*Sz + hy*Sy + hx*Sx

            W[1, 4] += Jz*Sz
            W[2, 4] += Jy*Sy
            W[3, 4] += Jx*Sx
            W[4, 4] += I

            return W

        heis_keys = ['XX', 'YY', 'ZZ', 'IX', 'IY', 'IZ']
        for key in self.strings.keys():
            if not (key in heis_keys or key[::-1] in heis_keys):
                print(key)
                raise NotImplementedError('only have mpos for heisenberg type hamiltonians')
            if self.strings.get(key[::-1], 0) != self.strings.get(key, 0):
                raise NotImplementedError('only have mpos for heisenberg type hamiltonians')
        return [heis_mpo(*[self.strings.get(key, 0) for key in heis_keys])]*L

    def to_matrices(self, L):
        h_i = self.to_matrix()
        h_0_strings = self.strings.copy()
        h_L_strings = self.strings.copy()
        for st in ['X', 'Y', 'Z']:
            h_0_strings.pop('I'+st, None)
            h_L_strings.pop(st+'I', None)
        H_0 = Hamiltonian(h_0_strings)._squeeze()
        H_L = Hamiltonian(h_L_strings)._squeeze()
        if L==2:
            return [h_i]
        elif L==3:
            return [H_0.to_matrix()] + [h_i]
        else:
            return [H_0.to_matrix()] + [h_i]*(L-3)+[H_L.to_matrix()]

    def from_matrix(self, mat):
        xyz = list(S.keys())
        strings = list(product(xyz, xyz))
        self.strings = {a+b:trace(kron(a, b)@mat) for a, b in strings}
        del self.strings['II']
        return self
