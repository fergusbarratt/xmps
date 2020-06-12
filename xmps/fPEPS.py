from functools import reduce
from itertools import product
import numpy as np
from scipy.linalg import norm
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


def tensor_grouping_diag(a, where):
    """where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
    turn a into a map from d, X1, X2 to X3, X4, where X1 is the virtual leg to anticlockwise of where.
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

def tensor_ungrouping_diag(a, where, shape):
    """undo tensor_grouping_diag
    where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
    \\   //
      4|1
      -A-
      3|2
    //   \\
    """
    ins = list(range(4))
    shape1 = [(np.array(shape[1:])*np.array([1] *
              (where-1)+[shape[0]]+[1]*(4-where)))[i] for i in ins[where-1:]+ins[:where-1]] # ungroup all but physical
    trans1 = ins[(4-(where-1)):]+ins[:(4-(where-1))] # undo cyclic permutation
    shape2 = shape[1:where]+(2,)+shape[where:]# ungroup physical
    trans2 = [where-1]+list(range(5))[:where-1]+list(range(5))[where:] #order correctly
    return a.reshape(*shape1).transpose(trans1).reshape(*shape2).transpose(trans2)

def tensor_isometrise_diag(tensor, where):
    """turn tensor into map from where to opposite (see diag below)
    where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
    \\   //
      4|1
      -A-
      3|2
    //   \\
    """
    shape = tensor.shape
    map = tensor_grouping_diag(tensor, where)
    in_s, out_s = map.shape
    if in_s >= out_s:
        print('c')
        b = np.linalg.qr(map.conj().T)[0]
        diff = np.abs(np.array(map.shape)-np.array(b.conj().T.shape))
        map = np.pad(b, list(zip([0]*len(diff), diff)))
    else:
        print('d')
        q = np.linalg.qr(map)[0]
        diff = np.abs(np.array(map.shape)-np.array(q.shape))
        map = np.pad(q, list(zip([0]*len(diff), diff)))
        #print(map@map.conj().T)
        #print(map.conj().T@map)

    tensor_2 = tensor_ungrouping_diag(map, where, shape)
    return tensor_2

def tensor_grouping_lrud(a, where):
    """turn tensor into map from where to opposite (see diag below)
    where = 1, 2, 3, 4 clockwise from top right. group physical leg with numbered leg, and map from leg + neighbours to single opposite leg
       1
       |
     4-A-2
       |
       3
    """
    ins = list(range(4))
    tens = group_physical(a, where)
    tens = tens.transpose(ins[where-1:]+ins[:where-1]).transpose([0, 1, 3, 2])
    return tens.reshape(np.prod(tens.shape[:3]), -1)

def tensor_ungrouping_lrud(a, where, shape):
    """undo tensor_grouping_diag
    where = 1, 2, 3, 4 clockwise from top right. group physical leg with leg to anticlockwise of number
       1
       |
     4-A-2
       |
       3
    """
    ins = list(range(4))
    shape1 = [(np.array(shape[1:])*np.array([1] *
              (where-1)+[shape[0]]+[1]*(4-where)))[i] for i in ins[where-1:]+ins[:where-1]] # ungroup all but physical
    shape1 = [shape1[0], shape1[1], shape1[3], shape1[2]]
    trans1 = ins[(4-(where-1)):]+ins[:(4-(where-1))] # undo cyclic permutation
    shape2 = shape[1:where]+(2,)+shape[where:]# ungroup physical
    trans2 = [where-1]+list(range(5))[:where-1]+list(range(5))[where:] #order correctly
    return a.reshape(*shape1).transpose([0, 1, 3, 2]).transpose(trans1).reshape(*shape2).transpose(trans2)

def tensor_isometrise_lrud(tensor, where):
    """turn tensor into isometry from where to opposite (see diag below)
    where = 1, 2, 3, 4 clockwise from top right. group neighbouring legs with where. 
       1
       |
     4-A-2
       |
       3
    """
    shape = tensor.shape
    map = tensor_grouping_lrud(tensor, where)
    in_s, out_s = map.shape
    if in_s >= out_s:
        print('a')
        b = np.linalg.qr(map.conj().T)[0]
        diff = np.abs(np.array(map.shape)-np.array(b.conj().T.shape))
        map = np.pad(b, list(zip([0]*len(diff), diff)))
        #print(c.shape, map.shape)
        #map = np.linalg.qr(map)[0]
        #print(map@map.conj().T)
        #print(map.conj().T@map)
        #print(map)
        #raise Exception
    else:
        print('b')
        q = np.linalg.qr(map)[0]
        diff = np.abs(np.array(map.shape)-np.array(q.shape))
        map = np.pad(q, list(zip([0]*len(diff), diff)))

    tensor_2 = tensor_ungrouping_lrud(map, where, shape)
    return tensor_2

def tensor_norm(tensor):
    return norm(tensor.transpose([1, 2, 3, 4, 0]))

def tensor_normalise(tensor):
    return tensor/tensor_norm(tensor)

def transfer_matrix(tensor):
    A = B = tensor
    return np.tensordot(A, B.conj(), [0, 0]).transpose(
                        [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
                        *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])

def test_tensor_reshaping(N):
    print('testing tensor reshaping ... ')
    for _ in range(N):
        for where in [2, 3, 1, 4]:
            where = 2
            A = np.random.randn(2, 2, 1, 1, 2)
            x = tensor_grouping_diag(A, where)
            q = tensor_ungrouping_diag(x, where, A.shape)
            assert np.allclose(A, q)

            x = tensor_grouping_lrud(A, where)
            q = tensor_ungrouping_lrud(x, where, A.shape)
            assert np.allclose(A, q)

            #x = tensor_isometrise_lrud(A, where)
            #assert where, np.allclose(x, tensor_isometrise_lrud(x, where))

            #x = tensor_isometrise_diag(A, where)
            #assert np.allclose(x, tensor_isometrise_diag(x, where))

            where = 3
            #A = np.random.randn(2, 2, 2, 1, 1)
            #x = tensor_isometrise_lrud(A, where)
            #print(ncon(transfer_matrix(x).reshape(2, 2, 4, 1, 1), [[1, 1, -1, -2, -3,]]))
            raise Exception

#test_tensor_reshaping(5)

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

    def rotate_90(self):
        #rotate peps clockwise 90
        new_data = [x.transpose([3, 0, 1, 2]) for x in self.data]
        return fMPO(new_data)

    def unrotate_90(self):
        new_data = [x.transpose([1, 2, 3, 0]) for x in self.data]
        return fMPO(new_data)

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

    def trivial(self, L):
        self.data = [np.ones(shape)
                     for shape in self.create_structure(L, 1, 1)]
        self.d = 1
        self.X = 1
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

    def isometrise(self, oc):
        """ specify oc as (x, y) where x is across and y is down from top left
        """
        c_i, c_j = self.oc = oc
        assert 0 <= c_i < self.Lx and 0 <= c_j < self.Ly
        for j, row in enumerate(self.data):
            for i, _ in enumerate(row):
                if i > c_i:
                    if j > c_j:
                        print('below right')
                        where = 2
                        self.data[j][i] = tensor_isometrise_diag(self.data[j][i], where)
                    elif j < c_j:
                        print('above right')
                        where = 1
                        self.data[j][i] = tensor_isometrise_diag(self.data[j][i], where)
                    elif j == c_j:
                        print('right')
                        where = 2
                        self.data[j][i] = tensor_isometrise_lrud(self.data[j][i], where)
                elif i < c_i:
                    if j > c_j:
                        print('below left')
                        where = 3
                        self.data[j][i] = tensor_isometrise_diag(self.data[j][i], where)
                    elif j < c_j:
                        print('above left')
                        where = 4
                        self.data[j][i] = tensor_isometrise_diag(self.data[j][i], where)
                    elif j == c_j:
                        print('left')
                        where = 4
                        self.data[j][i] = tensor_isometrise_lrud(self.data[j][i], where)
                elif i == c_i:
                    if j > c_j:
                        print('below')
                        where = 3
                        self.data[j][i] = tensor_isometrise_lrud(self.data[j][i], where)
                    elif j < c_j:
                        print('above')
                        where = 1
                        self.data[j][i] = tensor_isometrise_lrud(self.data[j][i], where)
                    elif j == c_j:
                        print('oc')
                        #self.data[j][i] = tensor_normalise(self.data[j][i])
        return self

    def create_structure(self, Lx, Ly, d, X):
        """
          0 1
           \|
          4-A-2
            |
            3

        """
        if Lx>1 and Ly>1:
            top = [[(d, 1, X, X, 1)]+[(d, 1, X, X, X) for _ in range(Lx-2)]+[(d, 1, 1, X, X)]]
            mid = [[(d, X, X, X, 1)]+[(d, X, X, X, X) for _ in range(Lx-2)]+[(d, X, 1, X, X)]
                   for _ in range(Ly-2)]
            bottom = [[(d, X, X, 1, 1)]+[(d, X, X, 1, X) for _ in range(Lx-2)]+[(d, X, 1, 1, X)]]

            return top+mid+bottom
        elif Lx==1 and Ly>1:
            return [[(d, 1, X, 1, 1)]]+[[(d, 1, X, 1, X)] for _ in range(Ly-2)]+[[(d, 1, 1, 1, X)]]
        elif Ly==1 and Lx>1:
            return [[(d, 1, 1, X, 1)]+[(d, X, 1, X, 1) for _ in range(Lx-2)]+[(d, X, 1, 1, 1)]]
        else:
            raise Exception

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

    def environment(self, site):
        # this is a mess. 
        i, j = site
        i, j = self.Lx-j-1, i
        # list of mpo rows
        # add a bunch of trivial boundary mpos to avoid having to do edges separately
        row_mpos = [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
                *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
                for A, B in zip(row, other_row)])
                for row, other_row in zip(self.data, self.data)]

        row_mpos = [fMPO().trivial(self.Lx)]+row_mpos+[fMPO().trivial(self.Lx)]
        above = reduce(lambda x, y: x*y, row_mpos[:j+1][::-1])
        above.data = [np.array([[[[1]]]])]+above.data+[np.array([[[[1]]]])]
        below = reduce(lambda x, y: x*y, row_mpos[j+2:][::-1])
        below.data = [np.array([[[[1]]]])]+below.data+[np.array([[[[1]]]])]
        same_row = row_mpos[j+1]
        same_row.data = [np.array([[[[1]]]])]+same_row.data+[np.array([[[[1]]]])]

        # lr -> ud, ud -> rl (rotate peps image 90 degs anticlockwise)
        col_mpos = [fMPO([above.data[k], same_row.data[k], below.data[k]]).unrotate_90() for k in range(self.Lx+2)][::-1] # leftmost column is now bottom row

        left = reduce(lambda x, y: x*y, col_mpos[:i+1][::-1])
        right = reduce(lambda x, y: x*y, col_mpos[i+2:][::-1])
        site_col = col_mpos[i+1]

        ring_data = left.data + [site_col.data[2]] + right.data[::-1] + [site_col.data[0]]
        con = [[-1, 2, 1, -2], [-3, 3, -13, 2], [-4, -5, 4, 3], [4, -6, 5, -14], [5, -7, -8, 6], [-15, 6, -9, 7], [8, 7, -10, -11], [1, -16, 8, -12]]
        env = ncon(ring_data, con)
        env = env.reshape(env.shape[-4:]).transpose([3, 0, 1, 2])
        return env

    def overlap(self, other):
        mpos = [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
                *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
                for A, B in zip(row, other_row)])
                for row, other_row in zip(self.data, other.data)]
        t = reduce(lambda x, y: x*y, mpos[::-1]).recombine()
        return t

    def tm(self, site):
        i, j = site
        A = B = self.data[i][j]
        return transfer_matrix(A)

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
    print('testing peps addition ... ')
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
    print('testing peps from mps ... ')
    np.set_printoptions(precision=1)
    for _ in range(N):
        x = fMPS().random(9, 2, 4).left_canonicalise()
        y = fPEPS().from_mps(x)
        assert np.allclose(x.recombine().reshape(-1), y.recombine('row_snake').reshape(-1))

def test_isometrize(N):
    L = 2
    for i, j in product(range(L), range(L)):
        i, j = 0, 0
        print('orthogonality center: ', i, j)
        x = fPEPS().random(L, L, 2, 2).isometrise((i, j))
        #print(x.tm((0, 1)))
        mpos = [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
                *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
                for A, B in zip(row, other_row)])
                for row, other_row in zip(x.data, x.data)]
        np.set_printoptions(4)
        print(*[x.recombine() for x in mpos], sep='\n\n')
        raise Exception
        


        tm = x.tm((i, j))
        env = x.environment((i, j))
        env_map = env.reshape(reduce(lambda x, y: x+y, [[int(np.sqrt(x)), int(np.sqrt(x))] for x in env.shape]))
        A = env_map[:, :, 0, 0, 0, 0, 0, 0]
        B = env_map[0, 0, :, :, 0, 0, 0, 0]
        C = env_map[0, 0, 0, 0, :, :, 0, 0]
        D = env_map[0, 0, 0, 0, 0, 0, :, :]
        print(env_map)
        print(A, B, C, D, sep='\n\n', end='\n\n')
        #print(env_map/ncon([A, B, C, D], [[-1, -2], [-3, -4], [-5, -6], [-7, -8]]))
        raise Exception
        # contract transfer matrix on site i, with environment on site i, and get the norm of x.
        assert np.allclose(np.real(ncon([tm, env], [[1, 2, 3, 4], [1, 2, 3, 4]])), x.full_overlap(x))
        print(x.full_overlap(x))
        #print(env)
        raise Exception

    for _ in range(N):
        for i, j in product(range(1), range(L)):
            print('orthogonality center: ', i, j)
            x = fPEPS().random(1, L, 2, 3).isometrise((i, j))
            assert np.allclose(x.full_overlap(x), 1)
        for i, j in product(range(L), range(1)):
            print('orthogonality center: ', i, j)
            x = fPEPS().random(L, 1, 2, 3).isometrise((i, j))
            assert np.allclose(x.full_overlap(x), 1)

def tests(N):
    test_fPEPS_from_mps(N)
    test_fPEPS_addition(N)
    test_fPEPS_evs(N)
    test_mpo_multiplication(N)

if __name__ == '__main__':
    test_isometrize(1)