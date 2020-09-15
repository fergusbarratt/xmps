import numpy as np
from itertools import product
from functools import reduce
from xmps.spin import spins
I = np.eye(2)
X, Y, Z = spins(0.5)
S = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

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

    def to_mpo(self, L):
        """to_mpo. 
           Only deals with Heisenberg type hamiltonians
           : i.e. if there are XY type terms they get ignored.
           also all 'YY' terms get ignored. 
           """
        Jx = self.strings['XX'] if 'XX' in self.strings else 0
        Jy = 0
        Jz = self.strings['ZZ'] if 'ZZ' in self.strings else 0
        hx = 2*self.strings['IX'] if 'IX' in self.strings else 0
        hy = 2*self.strings['IY'] if 'IY' in self.strings else 0
        hz = 2*self.strings['IZ'] if 'IZ' in self.strings else 0

        Id = np.eye(2, dtype = float)
        Sx = S['X']
        Sy = S['Y']
        Sz = S['Z']
        d = 2

        chi = 5
        W = np.zeros((chi, chi, d, d), dtype=np.complex128)
        W[0,0] += Id    
        W[0,1] += Sz
        W[0,2] += Sy 
        W[0,3] += Sx
        W[0,4] += hz*Sz + hy*Sy + hx*Sx
                
        W[1,4] += Jz*Sz 
        W[2,4] += Jy*Sy 
        W[3,4] += Jx*Sx
        W[4,4] += Id
        
        return L*[W]

    def to_matrix(self):
        assert self.strings is not None
        h_i = np.zeros((4, 4))+0j
        for js, J in self.strings.items():
            h_i += J*reduce(np.kron, [S[j] for j in js])
        self._matrix = h_i
        return h_i

    def to_finite_list(self, L):
        assert self.strings is not None
        h0_strings = self.strings.copy()
        hi_strings = self.strings.copy()
        for i in S.keys():
            if i+'I' in h0_strings:
                h0_strings[i+'I']*=2
                del hi_strings[i+'I']
            if 'I'+i in h0_strings:
                h0_strings['I'+i]*=2
                hi_strings['I'+i]*=2

        h0 = Hamiltonian(h0_strings).to_matrix()/L
        hi = Hamiltonian(hi_strings).to_matrix()/L
        return [h0]+(L-2)*[hi]

    def from_matrix(self, mat):
        xyz = list(S.keys())
        strings = list(product(xyz, xyz))
        self.strings = {a+b:np.trace(np.kron(S[a], S[b])@mat) for a, b in strings}
        del self.strings['II']
        return self
