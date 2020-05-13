from functools import reduce
import numpy as np
from xmps.fMPS import fMPS, ncon
from xmps.spin import paulis
X, Y, Z = paulis(0.5)
I = np.eye(2)


def to_diagram(tensor):
    string = """
              d={0} {1}
                 \|
                {4}-A-{2}
                  |
                  {3}""".format(*tensor.shape)
    return string


def tuple_to_diagram(shape):
    string = """
              d={0} {1}
                 \|
                {4}-A-{2}
                  |
                  {3}""".format(*shape)
    return string


def flatten(li):
    """n is the number of layers to flatten"""
    return reduce(lambda x, y: x+y, li)

def col_contract(a, b):
    # contracts two peps tensor on their column indices - (creates new peps tensor)
    ret = np.tensordot(a, b, [-2, -4]).transpose([0, 4, 1, 2, 5, 6, 3, 7]).reshape(a.shape[0]*b.shape[0], a.shape[1], a.shape[2]*b.shape[2], b.shape[3], a.shape[4]*b.shape[4])
    return ret
def row_contract(a, b):
    # contracts two peps tensor on their row indices - (creates new peps tensor)
    ret = np.tensordot(a, b, [-3, -1]).transpose([0, 4, 1, 5, 6, 2, 7, 3]).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1], b.shape[2], a.shape[3]*b.shape[3], a.shape[4])
    return ret

class fMPO(object):
    def __init__(self, data=None, d=None, X=None):
        if data is not None:
            self.L = len(data)
            if d is not None:
                self.d = d
            else:
                self.d = data[0].shape[0]
            self.X = X if X is not None else max(
                [max(x.shape[1:]) for x in data])
            self.data = data

    def __mul__(self, other):
        if isinstance(other, fMPS):
            other = fMPO().from_mps(other)

        if isinstance(other, fMPO):
            new_data = []
            d = self.d
            for W, W_ in zip(self.data, other.data):
                ncon_indices = [[1, -3, -4, -6], [-1, -2, 1, -5]]
                new_data.append(ncon([W, W_], ncon_indices).reshape(
                    W_.shape[0], W.shape[1]*W_.shape[1], W.shape[2], W_.shape[-1]*W.shape[-1]))

        return fMPO(new_data, d=d, X=self.X*other.X)

    def transpose_ud(self):
        self.data = [x.transpose([2, 1, 0, 3]) for x in self.data]
        return self

    def transpose_lr(self):
        self.data = [x.transpose([0, 3, 2, 1]) for x in self.data]
        return self

    def reverse(self):
        self.data = self.data[::-1]
        return self

    def create_structure(self, L, d, X):
        """
            0
            |
          3-A-1
            |
            2

        """
        return [(d, X, d, 1)]+[(d, X, d, X) for _ in range(L-2)] + [(d, 1, d, X)]

    def structure(self):
        return [x.shape for x in self.data]

    def __str__(self):
        return str(self.structure())

    def random(self, L, d, X):
        self.L, self.d, self.X = L, d, X
        self.data = [np.random.randn(*shape)
                     for shape in self.create_structure(L, d, X)]
        return self

    def from_mps(self, mps):
        self.data = [np.expand_dims(x.transpose([2, 0, 1]), 0) for x in mps]
        self.L = mps.L
        self.X = mps.D
        self.d = mps.d
        return self

    def recombine(self):
        ncon_indices = [[-1, 1, -self.L-1, self.L+1]]+[[-n-2, n+2, -self.L-2-n, n+1]
                                                       for n in range(self.L-2)] + [[-self.L, self.L+1, -2*self.L, self.L-1]]
        M = ncon(self.data, ncon_indices)
        M = M.reshape(np.prod(M.shape[:self.L]), np.prod(M.shape[self.L:]))
        return M.conj().T

class fPEPS(object):
    """finite PEPS:
    lists (Ly) of lists (Lx) of numpy arrays (rank 1, d) of numpy arrays (rank 4, X). Finite"""

    def __init__(self, data=None, d=None, X=None):
        if data is not None:
            self.Lx = len(data)
            self.Ly = len(data[0])
            if d is not None:
                self.d = d
            else:
                self.d = data[0][0].shape[0]
            self.X = X if X is not None else max(
                [max(x.shape[1:]) for x in flatten(data)])
            self.data = data

    def __str__(self):
        assert hasattr(self, 'Lx') and hasattr(self, 'Ly')
        return '\n'.join(map(str, self.structure()))

    def create_structure(self, Lx, Ly, d, X):
        """
          0 1
           \|
          4-A-2
            |
            3

        """
        top = [[(d, 1, X, X, 1)]+[(d, 1, X, X, X)
                                  for _ in range(Lx-2)]+[(d, 1, 1, X, X)]]
        mid = [[(d, X, X, X, 1)]+[(d, X, X, X, X) for _ in range(Lx-2)]+[(d, X, 1, X, X)]
               for _ in range(Ly-2)]
        bottom = [[(d, X, X, 1, 1)]+[(d, X, X, 1, X)
                                     for _ in range(Lx-2)]+[(d, X, 1, 1, X)]]

        return top+mid+bottom

    def structure(self):
        assert hasattr(self, 'Lx') and hasattr(
            self, 'Ly') and hasattr(self, 'd') and hasattr(self, 'X')
        return self.create_structure(self.Lx, self.Ly, self.d, self.X)

    def random(self, Lx, Ly, d, X):
        self.Lx = Lx
        self.Ly = Ly
        self.d = d
        self.X = X
        shapes = self.create_structure(Lx, Ly, d, X)
        self.data = [[np.random.randn(*shape)+1j*np.random.randn(*shape) for shape in row]
                     for row in shapes]
        return self

    def normalise(self):
        self.data[0][0] /= self.norm()
        return self

    def norm(self):
        return np.sqrt(np.abs(self.overlap(self)))

    def recombine(self):
        cols = []
        for row in self.data:
            cols.append(reduce(row_contract, row))
        res = reduce(col_contract, cols)
        return np.squeeze(res)

    def overlap(self, other):
        mpos = [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
                *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
                for A, B in zip(row, other_row)])
                for row, other_row in zip(self.data, other.data)]
        t = reduce(lambda x, y: x*y, mpos[::-1]).recombine()
        return t

    def apply(self, op, site):
        # indices go (across, down) from top left
        i, j = site
        self.data[j][i] = np.tensordot(op, self.data[j][i], [1, 0])
        return self

    def ev(self, op, site):
        return np.real(self.copy().apply(op, site).overlap(self))

    def copy(self):
        return fPEPS([[x.copy() for x in row] for row in self.data])

def test_mpo_multiplication(N):
    print('testing mpo mps multiplication ... ')
    for _ in range(N):
        A = fMPO().random(5, 2, 4)
        z = fMPO().from_mps(fMPS().random(5, 2, 2))

        A_ψ = A.recombine()
        z_ψ = z.recombine().reshape(-1)

        f1 = (A*z).recombine().reshape(-1)
        f2 = A_ψ@z_ψ
        assert np.allclose(f1, f2)
    print('testing mpo mpo multiplication ... ')
    for _ in range(N):
        A = fMPO().random(5, 2, 4)
        B = fMPO().random(5, 2, 6)

        A_ψ = A.recombine()
        B_ψ = B.recombine()

        f1 = (A*B).recombine()
        f2 = A_ψ@B_ψ
        assert np.allclose(f1, f2)


def test_fPEPS_evs(N):
    print('testing peps evs ... ')
    for _ in range(N):
        x = fPEPS().random(3, 3, 2, 2).normalise()
        e1 = x.ev(X, (0, 0))
        e2 = x.ev(Y, (0, 0))
        e3 = x.ev(Z, (0, 0))
        n = np.sqrt(e1**2+e2**2+e3**2)
        assert np.abs(e1) <= 1
        assert np.abs(e2) <= 1
        assert np.abs(e3) <= 1
        assert np.abs(n) <= 1

        x = fPEPS().random(3, 3, 2, 1).normalise()
        e1 = x.ev(X, (0, 0))
        e2 = x.ev(Y, (0, 0))
        e3 = x.ev(Z, (0, 0))
        n = np.sqrt(e1**2+e2**2+e3**2)
        assert np.abs(e1) <= 1
        assert np.abs(e2) <= 1
        assert np.abs(e3) <= 1
        assert np.allclose(n, 1)

        φ = x.recombine()
        XI = reduce(np.kron, [X]+[I]*8)
        YI = reduce(np.kron, [Y]+[I]*8)
        ZI = reduce(np.kron, [Z]+[I]*8)
        assert np.allclose(φ.conj().T@XI@φ, e1)
        assert np.allclose(φ.conj().T@YI@φ, e2)
        assert np.allclose(φ.conj().T@ZI@φ, e3)

        IXI = reduce(np.kron, [I]*3+[X]+[I]*5)
        assert np.allclose(x.ev(X, (0, 1)), φ.conj().T@IXI@φ)


if __name__ == '__main__':
    test_fPEPS_evs(3)
    test_mpo_multiplication(5)
