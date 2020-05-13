import numpy as np
from xmps.fMPS import fMPS, ncon

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
                [max(x.shape[1:]) for x in data])
            self.data = data

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
        mid = [[(d, X, X, X, 1)]+[(d, X, X, X, X) for _ in range(Lx-2)]+[(d, X, X, X, 1)]
               for _ in range(Ly-2)]
        bottom = [[(d, X, X, 1, 1)]+[(d, X, X, 1, X)
                                     for _ in range(Lx-2)]+[(d, X, 1, 1, X)]]

        return top+mid+bottom

    def structure(self):
        assert hasattr(self, 'Lx') and hasattr(
            self, 'Ly') and hasattr(self, 'd') and hasattr(self, 'X')
        return self.create_structure(self.Lx, self.Ly, self.d, self.X)

    def __str__(self):
        assert hasattr(self, 'Lx') and hasattr(self, 'Ly')
        return '\n'.join(map(str, self.structure()))

    def random(self, Lx, Ly, d, X):
        self.Lx = Lx
        self.Ly = Ly
        self.d = d
        self.X = X
        shapes = self.create_structure(Lx, Ly, d, X)
        self.data = [[np.random.randn(*shape) for shape in row] 
                      for row in shapes]
        return self

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
            assert self.d == other.d
            d = self.d
            for W, W_ in zip(self.data, other.data):
                ncon_indices = [[1, -3, -4, -6], [-1, -2, 1, -5]]
                new_data.append(ncon([W, W_], ncon_indices).reshape(W_.shape[0], W.shape[1]*W_.shape[1], W.shape[2], W_.shape[-1]*W.shape[-1]))

        return fMPO(new_data, d=d, X = self.X*other.X)

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
        self.data = [np.random.randn(*shape) for shape in self.create_structure(L, d, X)]
        return self

    def from_mps(self, mps):
        self.data = [np.expand_dims(x.transpose([2, 0, 1]), 0) for x in mps]
        self.L = mps.L
        self.X = mps.D
        self.d = mps.d
        return self

    def recombine(self):
        ncon_indices = [[-1, 1, -self.L-1, self.L+1]]+[[-n-2, n+2, -self.L-2-n, n+1] for n in range(self.L-2)] + [[-self.L, self.L+1, -2*self.L, self.L-1]]
        M = ncon(self.data, ncon_indices)
        M = M.reshape(np.prod(M.shape[:self.L]), np.prod(M.shape[self.L:]))
        return M



if __name__ == '__main__':
    A = fMPO().random(3, 2, 2)
    z = fMPO().from_mps(fMPS().random(3, 2, 2))

    A_ψ = A.recombine()
    z_ψ = z.recombine().reshape(-1)

    f1 = (A*z).recombine().reshape(-1)
    f2 = z_ψ@A_ψ
    print(f1-f2)
