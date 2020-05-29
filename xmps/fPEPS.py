from functools import reduce
import numpy as np
from xmps.fMPS import fMPS, ncon
from xmps.spin import paulis
from xmps.tensor import T
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
    ret = np.tensordot(a, b, [-2, -4]).transpose([0, 4, 1, 2, 5, 6, 3, 7]).reshape(
        a.shape[0]*b.shape[0], a.shape[1], a.shape[2]*b.shape[2], b.shape[3], a.shape[4]*b.shape[4])
    return ret


def row_contract(a, b):
    # contracts two peps tensor on their row indices - (creates new peps tensor)
    ret = np.tensordot(a, b, [-3, -1]).transpose([0, 4, 1, 5, 6, 2, 7, 3]).reshape(
        a.shape[0]*b.shape[0], a.shape[1]*b.shape[1], b.shape[2], a.shape[3]*b.shape[3], a.shape[4])
    return ret


def group_physical(a, where):
    """ where = 1, 2, 3, 4 clockwise from top. group physical leg with that virtual leg """
    assert 1 <= where <= 4
    ins = list(range(1, 5))
    trans = ins[:where-1]+[0]+ins[where-1:]
    shape = np.array(a.shape[1:])*np.array([1] *
                     (where-1)+[a.shape[0]]+[1]*(4-where))
    return a.transpose(trans).reshape(*shape)


def tensor_isometry(a, where):
    """where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
    \\   //
      4|1
      -A-
      3|2
    //   \\
    x[i:]+x[:i] cyclically permutes x i times.
    """
    ins = list(range(4))
    tens = group_physical(a, where)
    tens = tens.transpose(ins[where-1:]+ins[:where-1])
    return tens.reshape(np.prod(tens.shape[:2]), -1)

#def isometry_tensor(a, where, d=2):
#   """undo tensor_isometry
#   where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
#   \\   //
#     4|1
#     -A-
#     3|2
#   //   \\
#   x[i:]+x[:i] cyclically permutes x i times.
#   """
#   dX1X2, X3X4 = a.shape
#   #X, d = int(np.sqrt(XX)), int(dX1X2/d)
#   tens = a.reshape(d, X, X, X, X)
#   ins = list(range(1, 5))
#   return tens.transpose([0]+ins[5-where:]+ins[:5-where])


#A = np.random.randn(2, 1, 2, 2, 2)
#x = tensor_isometry(A, 4)
#B = isometry_tensor(x, 4)
#assert np.allclose(A, B)
#raise Exception

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

    def isometrize(self, oc):
        """ specify oc as (x, y) where x is across and y is down from top left
        """
        pass

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

    def add(self, other):
        new_data = []
        Lx, Ly = self.Lx, self.Ly
        for j, (row1, row2) in enumerate(zip(self.data, other.data)):
            new_data.append([])
            for i, (tens1, tens2) in enumerate(zip(row1, row2)):
                #if self.X>1 and other.X>1:
                #    new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, [0]+[x*int(x!=1) for x in list(tens2.shape[1:])])))+\
                #                        np.pad(tens2, list(zip([0]+[x*int(x!=1) for x in list(tens1.shape[1:])], [0]*tens1.ndim)))) 
                #    # add tensors block diagonally by padding with zeros with the shape of the other tensor
                #    # unless dim is one in that slot, then just add across that index. 
                #    # for mps this reduces to normal addition - works as long as X!= 1 anywhere
                #else: # annoying to have to do all this work just for X=1.
                assert Lx>1 and Ly>1
                #the below only works if Lx, Ly > 1 - need being on corner, edge, etc. to be mutually exclusive. Easy fix, can't be bothered
                if i == Lx-1: #right edge
                    if j == Ly-1: # bottom right corner [2, X, 1, 1, X]
                        pads1 = [0, tens2.shape[1], 0, 0, tens2.shape[4]]
                        pads2 = [0, tens1.shape[1], 0, 0, tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0: # top right corner [2, 1, 1, X, X]
                        pads1 = [0, 0, 0, tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, 0, 0, tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else: # right edge, not on corners [2, X, 1, X, X]
                        pads1 = [0, tens2.shape[1], 0, tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, tens1.shape[1], 0, tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                elif i == 0: #left_edge
                    if j == Ly-1: # bottom left corner, [2, X, X, 1, 1]
                        pads1 = [0, tens2.shape[1], tens2.shape[2], 0, 0]
                        pads2 = [0, tens1.shape[1], tens1.shape[2], 0, 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0: # top left corner, [2, 1, X, X, 1]
                        pads1 = [0, 0, tens2.shape[2], tens2.shape[3], 0]
                        pads2 = [0, 0, tens1.shape[2], tens1.shape[3], 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else: # left edge, not on corner, [2, X, X, X, 1]
                        pads1 = [0, tens2.shape[1], tens2.shape[2], tens2.shape[3], 0]
                        pads2 = [0, tens1.shape[1], tens1.shape[2], tens1.shape[3], 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                else:
                    if j == Ly-1: # bottom edge, not on corner, [2, X, X, 1, X]
                        pads1 = [0, tens2.shape[1], tens2.shape[2], 0, tens2.shape[4]]
                        pads2 = [0, tens1.shape[1], tens1.shape[2], 0, tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0: # top edge, not on corner, [2, 1, X, X, X]
                        pads1 = [0, 0, tens2.shape[2], tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, 0, tens1.shape[2], tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1)))+\
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else: # in bulk, [2, X, X, X, X]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, [0]+list(tens2.shape[1:]))))+\
                                            np.pad(tens2, list(zip([0]+list(tens1.shape[1:]), [0]*tens1.ndim))))

        return fPEPS(new_data)

    def from_mps(self, mps, Lx=None):
        """make an mps into a peps in row major snaking order """
        if Lx == None:
            Lx = np.sqrt(mps.L)
            assert np.allclose(int(Lx), Lx)
            Ly = Lx = int(Lx)
        else:
            Ly = mps.L/Lx
            assert np.allclose(int(Lx), Lx)
            Ly = int(Ly)
            Lx = int(Lx)

        reshaped = [mps[Lx*i:Lx*(i+1)] for i in range(Ly)]
        for j in range(Ly):
            # snake odd rows
            if j%2==1:
                reshaped[j] = reshaped[j][::-1]

        d = mps.d
        for i in range(Lx):
            for j in range(Ly):
                d, X1, X2 = reshaped[j][i].shape
                if j%2 == 0:
                    # going right
                    if i == Lx-1:
                        reshaped[j][i] = T(reshaped[j][i]).reshape(d, 1, 1, X2, X1)
                    elif i == 0:
                        reshaped[j][i] = reshaped[j][i].reshape(d, X1, X2, 1, 1)
                    else:
                        reshaped[j][i] = T(reshaped[j][i]).reshape(d, 1, X2, 1, X1)
                else:
                    # going left
                    if i == Lx-1:
                        reshaped[j][i] = reshaped[j][i].reshape(d, X1, 1, 1, X2)
                    elif i == 0:
                        reshaped[j][i] = reshaped[j][i].reshape(d, 1, X1, X2, 1)
                    else:
                        reshaped[j][i] = reshaped[j][i].reshape(d, 1, X1, 1, X2)

        return fPEPS(reshaped)

    def structure(self):
        return [[x.shape for x in row] for row in self.data]

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

    def recombine(self, qubit_ordering='reading'):
        if qubit_ordering == 'reading':
            cols = []
            for row in self.data:
                cols.append(reduce(row_contract, row))
            res = reduce(col_contract, cols)
            return np.squeeze(res)
        elif qubit_ordering == 'row_snake':
            cols = []
            for i, row in enumerate(self.data):
                if i%2==0:
                    cols.append(reduce(row_contract, row))
                else:
                    cols.append(reduce(row_contract, [x.transpose([0, 1, 4, 3, 2]) for x in row[::-1]]))
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

    def full_overlap(self, other):
        return np.abs(self.recombine().reshape(-1).conj().T@other.recombine().reshape(-1))

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

def test_fPEPS_addition(N):
    for _ in range(N):
        #x = fPEPS().random(3, 1, 2, 2).normalise()
        #y = fPEPS().random(3, 1, 2, 2).normalise()
        #assert np.allclose(x.recombine()+y.recombine(), x.add(y).recombine())

        x = fPEPS().random(3, 2, 2, 4).normalise()
        y = fPEPS().random(3, 2, 2, 2).normalise()
        assert np.allclose(x.recombine()+y.recombine(), x.add(y).recombine())

        x = fPEPS().random(3, 3, 2, 1).normalise()
        y = fPEPS().random(3, 3, 2, 1).normalise()
        assert np.allclose(x.recombine()+y.recombine(), x.add(y).recombine())

        x = fPEPS().random(3, 3, 2, 2).normalise()
        y = fPEPS().random(3, 3, 2, 2).normalise()
        assert np.allclose(x.recombine()+y.recombine(), x.add(y).recombine())

def test_fPEPS_from_mps(N):
    np.set_printoptions(precision=1)
    for _ in range(N):
        x = fMPS().random(9, 2, 4).left_canonicalise()
        y = fPEPS().from_mps(x)
        assert np.allclose(x.recombine().reshape(-1), y.recombine('row_snake').reshape(-1))

if __name__ == '__main__':
    test_fPEPS_from_mps(5)
    #test_fPEPS_addition(3)
    #test_fPEPS_evs(3)
    #test_mpo_multiplication(5)
