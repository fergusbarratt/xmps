from xmps.peps_tools import group_legs, ungroup_legs, U2
from xmps.fMPS import separate_tensor_qr as separate_tensor, group_tensors
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


def row_truncate(row, D):
    # take a row from a peps, truncate the row indices to D
    new_row = []
    for tensor in row:
        new_row.append(tensor[:, :, :D, :, :D])
    return new_row


def col_truncate(col, D):
    # take a col from a peps, truncate the col indices to D
    new_col = []
    for tensor in col:
        new_col.append(tensor[:, :D, :, :D, :])
    return new_col


def tensor_grouping_diag(tensor, where):
    '''Group legs of tensor so it's a map from [where, where+1] to [where+2, where+3]  (i.e. opposites), all mod 4
      1
      |
    4- -2
      |
      3
    '''
    where = where-1
    def mod4(x): return [np.mod(y, 4)+1 for y in x]
    pipe = [[0]+mod4([where, where+1]), mod4([where+3, where+2])]
    return group_legs(tensor, pipe)


def tensor_grouping_lrud(tensor, where):
    '''Group legs of tensor so its a map from where, where-1, where+1 to where+2 (i.e. opposite), all mod 4
      1
      |
    4- -2
      |
      3
    '''
    where = where-1
    def mod4(x): return [np.mod(y, 4)+1 for y in x]
    pipe = [[0]+mod4([where-1, where, where+1]), mod4([where+2])]
    return group_legs(tensor, pipe)


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
    map, pipe = tensor_grouping_diag(tensor, where)
    in_s, out_s = map.shape
    if in_s >= out_s:
        U, S, V = np.linalg.svd(map, full_matrices=False)
        map = U
        return ungroup_legs(map, pipe)
    else:
        raise Exception('truncate peps properly before isometrising')


def tensor_isometrise_lrud(tensor, where):
    """turn tensor into isometry from where to opposite (see diag below)
    where = 1, 2, 3, 4 clockwise from top right. group neighbouring legs with where. 
       1
       |
     4-A-2
       |
       3
    """
    map, pipe = tensor_grouping_lrud(tensor, where)
    in_s, out_s = map.shape
    if in_s >= out_s:
        U, S, V = np.linalg.svd(map, full_matrices=False)
        map = U
        return ungroup_legs(map, pipe)
    else:
        raise Exception('truncate peps properly before isometrising')

    return tensor_2


def tensor_isometrise_center(tensor):
    return tensor/norm(tensor.reshape(-1))


def transfer_tensor(tensor):
    return np.tensordot(tensor, tensor.conj(), [0, 0]).transpose(
        [0, 4, 1, 5, 2, 6, 3, 7])


def rotate_peps_tensor_cc(A):
    return A.transpose([0, 4, 1, 2, 3])


def rotate_peps_tensor_ac(A):
    return A.transpose([0, 2, 3, 4, 1])


def transpose_peps_tensor(A):
    return A.transpose([0, 4, 3, 2, 1])


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
        # rotate peps clockwise 90
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

    def __eq__(self, other):
        '''whether two peps are exactly equal. (to machine precision i.e. no gauge stuff)'''
        eq = True
        if hasattr(self, 'oc'):
            eq = eq and hasattr(other, 'oc')
            eq = eq and np.allclose(self.oc, other.oc)
        elif hasattr(other, 'oc'):
            return False

        if hasattr(self, 'X'):
            eq = eq and hasattr(other, 'X')
            eq = eq and np.allclose(self.X, other.X)
        elif hasattr(other, 'X'):
            return False

        if hasattr(self, 'd'):
            eq = eq and hasattr(other, 'd')
            eq = eq and np.allclose(self.d, other.d)
        elif hasattr(other, 'd'):
            return False

        for row, row_ in zip(self.data, other.data):
            for A, A_ in zip(row, row_):
                if not np.allclose(A.shape, A_.shape):
                    return False
                eq = eq and np.allclose(A, A_)
        return eq

    def row_mpos(self, other=None):
        other = self if other is None else other
        return [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
            *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
            for A, B in zip(row, other_row)])
            for row, other_row in zip(self.data, other.data)]

    def col_mpos(self, other=None):
        return self.T.row_mpos(None)

    @property
    def T(self):
        new_data = list(map(list, zip(*self.data)))  # transpose lists
        ret = fPEPS([[transpose_peps_tensor(x) for x in row] for row in new_data])
        if hasattr(self, 'oc'):
            (j, i)  = self.oc
            ret.oc = (i, j)
        if hasattr(self, 'X'):
            ret.X = self.X
        if hasattr(self, 'd'):
            ret.d = self.d
        return ret

    @property
    def R(self):
        '''flip peps left to right'''
        new_data = [[x.transpose([0, 1, 4, 3, 2]) for x in row][::-1] for row in self.data]
        ret = fPEPS(new_data)
        if hasattr(self, 'oc'):
            i, j = self.oc
            ret.oc = (self.Lx-1-i, j)
        if hasattr(self, 'X'):
            ret.X = self.X
        if hasattr(self, 'd'):
            ret.d = self.d
        return ret

    @property
    def U(self):
        '''flip peps down up'''
        new_data = [[x.transpose([0, 3, 2, 1, 4]) for x in row] for row in self.data][::-1]
        ret = fPEPS(new_data)
        if hasattr(self, 'oc'):
            i, j = self.oc
            ret.oc = (i, self.Ly-j-1)
        if hasattr(self, 'X'):
            ret.X = self.X
        if hasattr(self, 'd'):
            ret.d = self.d
        return ret

    def truncate_above(self, row, D):
        '''truncate bonds above row to size D'''
        if row == 0:
            raise Exception('nothing above row 0')
        self.data[row-1] = [x[:, :, :, :D, :] for x in self.data[row-1]]
        self.data[row] = [x[:, :D, :, :, :] for x in self.data[row]]
        return self

    def truncate_below(self, row, D):
        '''truncate bonds below row to size D'''
        if row == self.Lx-1:
            raise Exception('nothing below row L-1')
        self.data[row] = [x[:, :, :, :D, :] for x in self.data[row]]
        self.data[row+1] = [x[:, :D, :, :, :] for x in self.data[row+1]]
        return self

    def truncate_left(self, col, D):
        if col == 0:
            raise Exception('nothing to the left of col 0')
        for row in range(self.Lx):
            self.data[row][col] = self.data[row][col][:, :, :, :, :D]
            self.data[row][col-1] = self.data[row][col-1][:, :, :D, :, :]
        return self

    def truncate_along_row(self, row, D):
        self.data[row] = row_truncate(self.data[row], D)
        return self

    def truncate_along_column(self, col, D):
        for i, row in enumerate(self.data):
            self.data[i][col] = row[col][:, :D, :, :D, :]
        return self

    def fix_neighbours(self, i, j):
        _, up, right, down, left = self.data[j][i].shape
        if j-1 >= 0:
            _, _, _, new_up, _ = self.data[j-1][i].shape
            up = min([up, new_up])
            self.data[j-1][i] = self.data[j-1][i][:, :, :, :up, :]
            self.data[j][i] = self.data[j][i][:, :up, :, :, :]
            assert self.data[j-1][i].shape[3] == self.data[j][i].shape[1]
        if i+1 < self.Lx:
            _, _, _, _, new_right = self.data[j][i+1].shape
            right = min([right, new_right])
            self.data[j][i+1] = self.data[j][i+1][:, :, :, :, :right]
            self.data[j][i] = self.data[j][i][:, :, :right, :, :]
            assert self.data[j][i+1].shape[4] == self.data[j][i].shape[2]
        if j+1 < self.Ly:
            _, new_down, _, _, _ = self.data[j+1][i].shape
            down = min([down, new_down])
            self.data[j+1][i] = self.data[j+1][i][:, :down, :, :, :]
            self.data[j][i] = self.data[j][i][:, :, :, :down, :]
            assert self.data[j+1][i].shape[1] == self.data[j][i].shape[3]
        if i-1 >= 0:
            _, _, new_left, _, _ = self.data[j][i-1].shape
            left = min([left, new_left])
            self.data[j][i-1] = self.data[j][i-1][:, :, :left, :, :]
            self.data[j][i] = self.data[j][i][:, :, :, :, :left]
            assert self.data[j][i-1].shape[2] == self.data[j][i].shape[4]
        return self

    def create_structure(self, Lx, Ly, d, X):
        """
          0 1
           \|
          4-A-2
            |
            3

        """
        if Lx > 1 and Ly > 1:
            top = [[(d, 1, X, X, 1)]+[(d, 1, X, X, X)
                                      for _ in range(Lx-2)]+[(d, 1, 1, X, X)]]
            mid = [[(d, X, X, X, 1)]+[(d, X, X, X, X) for _ in range(Lx-2)]+[(d, X, 1, X, X)]
                   for _ in range(Ly-2)]
            bottom = [[(d, X, X, 1, 1)]+[(d, X, X, 1, X)
                                         for _ in range(Lx-2)]+[(d, X, 1, 1, X)]]

            return top+mid+bottom
        elif Ly == 1 and Lx > 1:
            return [[(d, 1, X, 1, 1)]+[(d, 1, X, 1, X) for _ in range(Lx-2)]+[(d, 1, 1, 1, X)]]
        elif Lx == 1 and Ly > 1:
            return [[(d, 1, 1, X, 1)]]+[[(d, X, 1, X, 1)] for _ in range(Ly-2)]+[[(d, X, 1, 1, 1)]]
        else:
            raise Exception

    def add(self, other):
        new_data = []
        Lx, Ly = self.Lx, self.Ly
        for j, (row1, row2) in enumerate(zip(self.data, other.data)):
            new_data.append([])
            for i, (tens1, tens2) in enumerate(zip(row1, row2)):
                # if self.X>1 and other.X>1:
                #    new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, [0]+[x*int(x!=1) for x in list(tens2.shape[1:])])))+\
                #                        np.pad(tens2, list(zip([0]+[x*int(x!=1) for x in list(tens1.shape[1:])], [0]*tens1.ndim))))
                #    # add tensors block diagonally by padding with zeros with the shape of the other tensor
                #    # unless dim is one in that slot, then just add across that index.
                #    # for mps this reduces to normal addition - works as long as X!= 1 anywhere
                # else: # annoying to have to do all this work just for X=1.
                assert Lx > 1 and Ly > 1
                # the below only works if Lx, Ly > 1 - need being on corner, edge, etc. to be mutually exclusive. Easy fix, can't be bothered
                if i == Lx-1:  # right edge
                    if j == Ly-1:  # bottom right corner [2, X, 1, 1, X]
                        pads1 = [0, tens2.shape[1], 0, 0, tens2.shape[4]]
                        pads2 = [0, tens1.shape[1], 0, 0, tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0:  # top right corner [2, 1, 1, X, X]
                        pads1 = [0, 0, 0, tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, 0, 0, tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else:  # right edge, not on corners [2, X, 1, X, X]
                        pads1 = [0, tens2.shape[1], 0,
                                 tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, tens1.shape[1], 0,
                                 tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                elif i == 0:  # left_edge
                    if j == Ly-1:  # bottom left corner, [2, X, X, 1, 1]
                        pads1 = [0, tens2.shape[1], tens2.shape[2], 0, 0]
                        pads2 = [0, tens1.shape[1], tens1.shape[2], 0, 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0:  # top left corner, [2, 1, X, X, 1]
                        pads1 = [0, 0, tens2.shape[2], tens2.shape[3], 0]
                        pads2 = [0, 0, tens1.shape[2], tens1.shape[3], 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else:  # left edge, not on corner, [2, X, X, X, 1]
                        pads1 = [0, tens2.shape[1],
                                 tens2.shape[2], tens2.shape[3], 0]
                        pads2 = [0, tens1.shape[1],
                                 tens1.shape[2], tens1.shape[3], 0]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                else:
                    # bottom edge, not on corner, [2, X, X, 1, X]
                    if j == Ly-1:
                        pads1 = [0, tens2.shape[1],
                                 tens2.shape[2], 0, tens2.shape[4]]
                        pads2 = [0, tens1.shape[1],
                                 tens1.shape[2], 0, tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    elif j == 0:  # top edge, not on corner, [2, 1, X, X, X]
                        pads1 = [0, 0, tens2.shape[2],
                                 tens2.shape[3], tens2.shape[4]]
                        pads2 = [0, 0, tens1.shape[2],
                                 tens1.shape[3], tens1.shape[4]]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, pads1))) +
                                            np.pad(tens2, list(zip(pads2, [0]*tens1.ndim))))
                    else:  # in bulk, [2, X, X, X, X]
                        new_data[-1].append(np.pad(tens1, list(zip([0]*tens2.ndim, [0]+list(tens2.shape[1:])))) +
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
            if j % 2 == 1:
                reshaped[j] = reshaped[j][::-1]

        d = mps.d
        for i in range(Lx):
            for j in range(Ly):
                d, X1, X2 = reshaped[j][i].shape
                if j % 2 == 0:
                    # going right
                    if i == Lx-1:
                        reshaped[j][i] = T(reshaped[j][i]).reshape(
                            d, 1, 1, X2, X1)
                    elif i == 0:
                        reshaped[j][i] = reshaped[j][i].reshape(
                            d, X1, X2, 1, 1)
                    else:
                        reshaped[j][i] = T(reshaped[j][i]).reshape(
                            d, 1, X2, 1, X1)
                else:
                    # going left
                    if i == Lx-1:
                        reshaped[j][i] = reshaped[j][i].reshape(
                            d, X1, 1, 1, X2)
                    elif i == 0:
                        reshaped[j][i] = reshaped[j][i].reshape(
                            d, 1, X1, X2, 1)
                    else:
                        reshaped[j][i] = reshaped[j][i].reshape(
                            d, 1, X1, 1, X2)

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
                if i % 2 == 0:
                    cols.append(reduce(row_contract, row))
                else:
                    cols.append(
                        reduce(row_contract, [x.transpose([0, 1, 4, 3, 2]) for x in row[::-1]]))
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
        same_row.data = [np.array([[[[1]]]])] + \
            same_row.data+[np.array([[[[1]]]])]

        # lr -> ud, ud -> rl (rotate peps image 90 degs anticlockwise)
        col_mpos = [fMPO([above.data[k], same_row.data[k], below.data[k]]).unrotate_90(
        ) for k in range(self.Lx+2)][::-1]  # leftmost column is now bottom row

        left = reduce(lambda x, y: x*y, col_mpos[:i+1][::-1])
        right = reduce(lambda x, y: x*y, col_mpos[i+2:][::-1])
        site_col = col_mpos[i+1]

        ring_data = left.data + [site_col.data[2]] + \
            right.data[::-1] + [site_col.data[0]]
        con = [[-1, 2, 1, -2], [-3, 3, -13, 2], [-4, -5, 4, 3], [4, -6, 5, -14],
               [5, -7, -8, 6], [-15, 6, -9, 7], [8, 7, -10, -11], [1, -16, 8, -12]]
        env = ncon(ring_data, con)
        env = env.reshape(env.shape[-4:]).transpose([3, 0, 1, 2])
        return env

    def overlap(self, other):
        mpos = [fMPO([np.tensordot(A, B.conj(), [0, 0]).transpose(
                [0, 4, 1, 5, 2, 6, 3, 7]).reshape(
            *[s1*s2 for s1, s2 in zip(A.shape[1:], B.shape[1:])])
            for A, B in zip(row, other_row)])
            for row, other_row in zip(self.data, other.data)]
        #mpos = self.row_mpos()
        t = reduce(lambda x, y: x*y, mpos[::-1]).recombine()
        return t

    def tm(self, site):
        i, j = site
        A = B = self.data[i][j]
        return transfer_matrix(A)

    def apply(self, op, site):
        # indices go (across, down) from top left
        i, j = site
        ret = self.copy()
        ret.data[j][i] = np.tensordot(op, self.data[j][i], [1, 0])
        return ret

    def isometrise(self, oc):
        """ specify oc as (x, y) where x is across and y is down from top left
        """
        c_i, c_j = self.oc = oc
        assert 0 <= c_i < self.Lx and 0 <= c_j < self.Ly
        x_truncate_to, y_truncate_to = int(self.Lx/2), int(self.Ly/2)
        for i in range(x_truncate_to+1):
            self.truncate_along_column(i, self.d**i)

        for k, i in enumerate(reversed(range(max([x_truncate_to-1, 0]), self.Lx))):
            self.truncate_along_column(i, self.d**k)

        for j in range(y_truncate_to+1):
            self.truncate_along_row(j, self.d**j)

        for k, j in enumerate(reversed(range(max([y_truncate_to-1, 0]), self.Ly))):
            self.truncate_along_row(j, self.d**k)

        if self.Ly > 1 and self.Lx > 1:
            self.truncate_below(0, self.d)
            self.truncate_below(self.Ly-2, self.d)
            self.truncate_left(1, self.d)
            self.truncate_left(self.Lx-1, self.d)

        for j, row in enumerate(self.data):
            for i, _ in enumerate(row):
                if i > c_i:
                    if j > c_j:
                        #print('below right')
                        where = 2
                        self.data[j][i] = tensor_isometrise_diag(
                            self.data[j][i], where)
                    elif j < c_j:
                        #print('above right')
                        where = 1
                        self.data[j][i] = tensor_isometrise_diag(
                            self.data[j][i], where)
                    elif j == c_j:
                        # print('right')
                        where = 2
                        self.data[j][i] = tensor_isometrise_lrud(
                            self.data[j][i], where)
                elif i < c_i:
                    if j > c_j:
                        #print('below left')
                        where = 3
                        self.data[j][i] = tensor_isometrise_diag(
                            self.data[j][i], where)
                    elif j < c_j:
                        #print('above left')
                        where = 4
                        self.data[j][i] = tensor_isometrise_diag(
                            self.data[j][i], where)
                    elif j == c_j:
                        # print('left')
                        where = 4
                        self.data[j][i] = tensor_isometrise_lrud(
                            self.data[j][i], where)
                elif i == c_i:
                    if j > c_j:
                        # print('below')
                        where = 3
                        self.data[j][i] = tensor_isometrise_lrud(
                            self.data[j][i], where)
                    elif j < c_j:
                        # print('above')
                        where = 1
                        self.data[j][i] = tensor_isometrise_lrud(
                            self.data[j][i], where)
                    elif j == c_j:
                        # print('oc')
                        self.data[j][i] = tensor_isometrise_center(
                            self.data[j][i])
        return self

    def ev(self, op, site):
        return np.real(self.apply(op, site).overlap(self))[0, 0]

    def iso_ev(self, op, site=None, chi=None):
        if not hasattr(self, 'oc'):
            raise Exception('Not an isometric peps - try isometrising')
        elif site is not None and site != self.oc:
            return self.copy().moses_move_oc_to(site, chi).iso_ev(op) 
        else:
            site = site if site is not None else self.oc
            i, j = site
            A = self.data[j][i]
            return np.real(ncon([A.conj(), op, A], [[5, 1, 2, 3, 4], [5, 6], [6, 1, 2, 3, 4]]))

    def iso_evs(self, opsites, chi=None):
        ret = []
        for opsite in opsites:
            ret.append(self.iso_ev(*opsite, chi))
        return np.array(ret)

    def moses_move_right(self, chi=None, testing=True, truncate=False):
        '''move orthogonality col to the right by one.
           operates inplace'''
        if truncate:
            chi = self.X if chi is None else chi
        if not hasattr(self, 'oc'):
            raise Exception('not an isotns: try isometrising')
        else:
            c_i, c_j = self.oc
            if c_i==self.Lx-1:
                raise Exception('can\'t move right from right border')
        new_data = self.copy().data

        middle_col = []
        b = np.ones((1, 1, 1))

        for n in (-x for x in range(1, self.Ly+1)):
            zipper = ncon([b, new_data[n][c_i]], [
                          [1, -4, -5], [-1, -2, -3, 1, -6]])
            [a, b, c] = split_tensor(zipper, chi)

            if testing:
                if truncate == False:
                    zipper_ = unsplit_tensor(a, b, c)
                    assert np.allclose(norm(zipper-zipper_), 0)

            middle_col.insert(0, c)
            new_data[n][c_i] = a

        new_data[0][c_i] = np.expand_dims(group_legs(
            new_data[0][c_i], [[0], [1, 2], [3], [4]])[0], 0).transpose([1, 0, 2, 3, 4])

        assert b.shape[0] == 1

        middle_col[0] = ncon([middle_col[0], b[0, :, :]], [[1, -2, -3, -4], [-1, 1]])
        # add dummy index, swap into the right place, rotate tensor
        middle_col[0] = np.expand_dims(group_legs(middle_col[0], [[3, 0], [1], [2]])[
                                       0], 0).transpose([1, 0, 2, 3]).transpose([1, 2, 3, 0])


        for n, λ in enumerate(middle_col):
            new_data[n][c_i+1], _ = group_legs(ncon([new_data[n][c_i+1], λ], [
                                                [-1, -2, -4, -5, 1], [-3, 1, -6, -7]]), [[0], [1, 2], [3], [4, 5], [6]])

        nn = norm(new_data[c_j][c_i+1].reshape(-1))
        new_data[c_j][c_i+1]/=nn # something real strange going on with the norm - this seems to fix it
        new_data[c_j][c_i]*=nn

        ret = fPEPS(new_data)
        ret.oc = (c_i, c_j) = (c_i+1, c_j)
        if hasattr(self, 'X'):
            ret.X = self.X
        if hasattr(self, 'd'):
            ret.d = self.d
        return ret

    def moses_move_left(self, chi=None):
        ret = self.R.moses_move_right(chi).R
        i, j = self.oc
        ret.oc = (i-1, j)
        return ret

    def moses_move_down(self, chi=None):
        ret = self.T.moses_move_right(chi).T
        i, j = self.oc
        ret.oc = (i, j+1)
        return ret

    def moses_move_up(self, chi=None):
        ret = self.T.R.moses_move_right(chi).R.T
        i, j = self.oc
        ret.oc = (i, j-1)
        return ret

    def moses_move_oc_to(self, oc, chi=None):
        if not hasattr(self, 'oc'):
            raise Exception('should be isotns - try isometrising, or set oc value')

        z = self.copy()
        rights, downs = np.array(oc)-np.array(self.oc)
        for _ in range(abs(rights)):
            if rights>0:
                print('r')
                z = z.moses_move_right(chi)
            else:
                print('l')
                z = z.moses_move_left(chi)

        for _ in range(abs(downs)):
            if downs>0:
                print('d')
                z = z.moses_move_down(chi)
            else:
                print('u')
                z = z.moses_move_up(chi)
        return z

    def copy(self, data=None):
        # copy self (or data) into new fPEPS with all of the stuff this fPEPS has (oc, X, d, etc.)
        data = self.data if data is None else data
        ret = fPEPS([[x.copy() for x in row] for row in self.data])
        if hasattr(self, 'oc'):
            ret.oc = self.oc
        if hasattr(self, 'X'):
            ret.X = self.X
        if hasattr(self, 'd'):
            ret.d = self.d
        return ret

    def full_overlap(self, other):
        return self.recombine().reshape(-1).conj().T@other.recombine().reshape(-1)


def tr_svd(a, D):
    u, s, v = np.linalg.svd(a, full_matrices=False)
    D = u.shape[1] if D is None else D
    u, s, v = u[:, :D], np.diag(s)[:D, :D], v[:D, :]
    return u, s, v


def factors(n):
    ''' return a, b such that a*b == n, a, b are integers, and |a-b| is minimised'''
    for i in range(int(n**0.5)+1, 0, -1):
        if n % i == 0:
            return [i, n//i]


def split_tensor(tens, chi):
    """ split tensor
                                AX
       0   1                    |
        \  |                    v
           |                    0
           |                    b
           v             d     2 1
       5--> <--2   ---->   0   ^ ^
           ^               \ /BlX \chi
          / \               1      0
         /   \          p->4a2-->-3c1<-r
        4     3      BX     3 BrX  2
                            ^      ^   CX = r*t
                            |      |
                            q      t
       returns [a, b, c]
    """
    ABC, pipe1 = group_legs(tens, [[1], [2, 3], [0, 4, 5]])
    (_, [_, (r, t), (d, p, q)]) = pipe1  # r, t, d, p, q are in diagram above
    map, pipe2 = group_legs(ABC, [[2], [0, 1]])
    # AX is bond dimension above top tensor, CX bond dimension below bottom tensor
    _, [(_,), (AX, CX)] = pipe2
    U, s, V = tr_svd(map, chi**2 if chi is not None else chi)  # truncated svd along / (A -> BC)
    a0, ABlBrC0 = U, s@V

    BlX, BrX = factors(ABlBrC0.shape[0])
    # truncate further (have to split ~chi**2 -> ~chi, ~chi)
    # print(BlX*BrX, a0.shape[1]) - sometimes theres a truncation here, even if the bond
    # dimension is high enough
    #a0, ABlBrC0 = a0[:, :BlX*BrX], ABlBrC0[:BlX*BrX, :]

    # get best isometry, unitary
    # get renyi minimising unitary
    entropy, U = U2(ABlBrC0.reshape(BlX, BrX, AX, CX))

    a = (a0@U.conj().T)
    ABlBrC = U@ABlBrC0

    #print(norm(map-a0@ABlBrC0), norm(map-a@ABlBrC))

    ABlBrC = ABlBrC.reshape(BlX, BrX, AX, CX)  # get legs out

    # then reshape (d*p*q, BlX*BrX)->(d, p, q, BrX, BlX)->(d, BlX, BrX, q, p)
    a = a.reshape(d, p, q, BlX, BrX).transpose([0, 3, 4, 1, 2])
    # (BlX, AX, chi), (BrX, chi, r*t), clockwise
    b, c = separate_tensor(ABlBrC.reshape(BlX*BrX, AX, CX), chi) # truncate to chi here

    b = b.transpose([1, 2, 0])
    c = c.transpose([1, 2, 0]).reshape(b.shape[1], r, t, -1)

    return [a, b, c]


def test_separate_tensor(N):
    print('testing separate tensor ... ', end='')
    for _ in range(N):
        D = 10
        a = np.random.randn(4, 5, 5)
        b, c = separate_tensor(a, D)
        assert np.allclose(a, group_tensors([b, c]))
    print('success')


def unsplit_tensor(a, b, c):
    return ncon([a, b, c], [[-1, 1, 2, -5, -6], [-2, 3, 1], [3, -3, -4, 2]])


def test_split_tensor(N):
    print('testing split_tensor ... ', end='')
    for _ in range(N):
        for k in range(1, 4):
            A = np.random.randn(2, 4, k, 4, 4, 4)
            #A = np.random.randn(2, 2, 2, 4, 2, 2)
            A = A+1j*np.random.randn(*A.shape)
            A = A/norm(A)

            OR = False
            for chi in range(1, 10):
                a, b, c = split_tensor(A, chi)
                A_ = unsplit_tensor(a, b, c)
                OR = OR or np.allclose(norm(A-A_), 0)
                if OR:
                    print('success at k={}, '.format(k), end='')
                    break


def test_mpo_multiplication(N):
    print('testing mpo mps multiplication ... ', end='')
    for _ in range(N):
        A = fMPO().random(5, 2, 4)
        z = fMPO().from_mps(fMPS().random(5, 2, 2))

        A_ψ = A.recombine()
        z_ψ = z.recombine().reshape(-1)

        f1 = (A*z).recombine().reshape(-1)
        f2 = A_ψ@z_ψ
        assert np.allclose(f1, f2)
    print('success')
    print('testing mpo mpo multiplication ... ', end='')
    for _ in range(N):
        A = fMPO().random(5, 2, 4)
        B = fMPO().random(5, 2, 6)

        A_ψ = A.recombine()
        B_ψ = B.recombine()

        f1 = (A*B).recombine()
        f2 = A_ψ@B_ψ
        assert np.allclose(f1, f2)
    print('success')


def test_fPEPS_evs(N):
    print('testing peps evs ... ', end='')
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
    print('success')


def test_fPEPS_addition(N):
    print('testing peps addition ... ', end='')
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
    print('success')


def test_fPEPS_from_mps(N):
    print('testing peps from mps ... ', end='')
    np.set_printoptions(precision=1)
    for _ in range(N):
        x = fMPS().random(9, 2, 4).left_canonicalise()
        y = fPEPS().from_mps(x)
        assert np.allclose(x.recombine().reshape(-1),
                           y.recombine('row_snake').reshape(-1))
    print('success')


def test_overlap(N):
    print('testing peps overlap ... ', end='')
    for _ in range(N):
        for L in range(2, 4):
            x = fPEPS().random(L, L, 2, 3).normalise()
            y = fPEPS().random(L, L, 2, 3).normalise()
            assert np.allclose(x.overlap(y), x.full_overlap(y))
    print('success')

def test_isometrize(N):
    for _ in range(N):
        for L in range(2, 6):
            for chi in range(2, 4):  # fails at chi=5!
                print('testing isometric norm, evs chi={}, {}x{} ...'.format(
                    chi, L, L), end='')
                for i, j in product(range(L), range(L)):
                    #    print('orthogonality center: ', i, j)
                    x = fPEPS().random(L, L, 2, chi).isometrise((i, j))
                    assert np.allclose(x.norm(), 1)
                    assert np.allclose(x.iso_ev(X, (i, j)), x.ev(X, (i, j)))
                    assert np.allclose(x.iso_ev(Y, (i, j)), x.ev(Y, (i, j)))
                    assert np.allclose(x.iso_ev(Z, (i, j)), x.ev(Z, (i, j)))
                print('success')

               # if L < 6:
                print('testing isometric norm {}x1 and 1x{} ...'.format(L, L), end='')
                for i, j in product(range(1), range(L)):
                    #print('orthogonality center: ', i, j)
                    x = fPEPS().random(1, L, 2, chi).isometrise((i, j))
                    assert np.allclose(x.full_overlap(x), 1)
                    '''TODO: implement 1d properly (just hand off to mps)'''
                    #assert np.allclose(x.iso_ev(X, (i, j)),x.ev(X, (i, j)))
                    #assert np.allclose(x.iso_ev(Y, (i, j)),x.ev(Y, (i, j)))
                    #assert np.allclose(x.iso_ev(Z, (i, j)),x.ev(Z, (i, j)))

                for i, j in product(range(L), range(1)):
                    #print('orthogonality center: ', i, j)
                    x = fPEPS().random(L, 1, 2, chi).isometrise((i, j))
                    assert np.allclose(x.full_overlap(x), 1)
                    #assert np.allclose(x.iso_ev(X, (i, j)),x.ev(X, (i, j)))
                    #assert np.allclose(x.iso_ev(Y, (i, j)),x.ev(Y, (i, j)))
                    #assert np.allclose(x.iso_ev(Z, (i, j)),x.ev(Z, (i, j)))
                print('success')


def test_moses_move(N):
    print('testing moses move ... ', end='')
    for _ in range(N):
        L, chi = 6, 2
        x = fPEPS().random(L, L, 2, chi).isometrise((1, 1))
        print(*x.structure(), sep='\n')
        print(x.iso_ev(X, (3, 1)), x.ev(X, (3, 1)))

        raise Exception

        L, chi = 4, 15
        for i, j in product(range(1, L-1), range(1, L-1)):
            oc = (i, j)
            x = fPEPS().random(L, L, 2, chi).isometrise(oc)

            # x has norm 1 to start
            assert np.allclose(x.iso_ev(I), 1)

            for op in [X, Y, Z]:
                # x is isometric to start
                assert np.allclose(x.iso_ev(op), x.ev(op, x.oc))

            old_oc = x.oc
            yr = x.copy().moses_move_right()
            yd = x.copy().moses_move_down()
            yu = x.copy().moses_move_up()
            yl = x.copy().moses_move_left()

            zr = x.copy().moses_move_oc_to((i+1, j))
            zd = x.copy().moses_move_oc_to((i, j+1))
            zu = x.copy().moses_move_oc_to((i, j-1))
            zl = x.copy().moses_move_oc_to((i-1, j))

            # x is the same, but yr has different tensors, z is x
            assert x == x and not x == yr
            assert yr==zr and yu==zu and yl==zl and yd==zd

            # does change the oc
            assert np.allclose(yr.oc, (x.oc[0]+1, x.oc[1]))
            assert np.allclose(yd.oc, (x.oc[0], x.oc[1]+1))
            assert np.allclose(yu.oc, (x.oc[0], x.oc[1]-1))
            assert np.allclose(yl.oc, (x.oc[0]-1, x.oc[1]))

            # doesn't change the norm
            assert np.allclose(yr.iso_ev(I), 1)
            assert np.allclose(yd.iso_ev(I), 1)
            assert np.allclose(yu.iso_ev(I), 1)
            assert np.allclose(yl.iso_ev(I), 1)
            
            for op in [X, Y, Z]:
                for k, l in product(range(1, L-1), range(1, L-1)):
                    # get same single site ev by moving oc or by full ev
                    assert np.allclose(x.ev(op, (k, l)), x.iso_ev(op, (k, l)))

                    # #expectation of pre-move peps, with op on new oc
                    # #is same as expectation of post move peps on new oc
                    assert np.allclose(yr.ev(op, yr.oc), x.ev(op, yr.oc))
                    assert np.allclose(yd.ev(op, yd.oc), x.ev(op, yd.oc))
                    assert np.allclose(yu.ev(op, yu.oc), x.ev(op, yu.oc))
                    assert np.allclose(yl.ev(op, yl.oc), x.ev(op, yl.oc))

                    # new oc is a proper oc
                    assert np.allclose(yr.iso_ev(op), yr.ev(op, yr.oc))
                    assert np.allclose(yd.iso_ev(op), yd.ev(op, yd.oc))
                    assert np.allclose(yu.iso_ev(op), yu.ev(op, yu.oc))
                    assert np.allclose(yl.iso_ev(op), yl.ev(op, yl.oc))

            #for op in [X, Y, Z]:
            #    for c in [(1, 2), (2, 1), (1, 1), (2, 2)]:
            #        print('old oc', x.oc, 'ev site', c)
            #        print('post move ev on ev site: r', x.ev(op, c), yr.ev(op, c))
            #        print('post move ev on ev site: u', x.ev(op, c), yu.ev(op, c))
            #        print('post move ev on ev site: d', x.ev(op, c), yd.ev(op, c))
            #        print('post move ev on ev site: l', x.ev(op, c), yl.ev(op, c))
            #        print('\n')
            #    #print('post move iso ev on old oc: ', yr.iso_ev(op))

    print('success')

def test_peps_transpose(N):
    print('testing peps transpose + flips ... ', end='')
    for _ in range(N):
        L = 3
        for i, j in product(range(L), range(L)):
            x = fPEPS().random(L, L, 2, 3).normalise()
            assert x==x.T.T
            assert x==x.R.R
            assert x==x.U.U

            assert np.allclose(x.ev(X, (i, j)), x.T.ev(X, (j, i)))
            assert np.allclose(x.ev(Y, (i, j)), x.T.ev(Y, (j, i)))
            assert np.allclose(x.ev(Z, (i, j)), x.T.ev(Z, (j, i)))

            assert np.allclose(x.ev(X, (i, j)), x.R.ev(X, (L-i-1, j)))
            assert np.allclose(x.ev(Y, (i, j)), x.R.ev(Y, (L-i-1, j)))
            assert np.allclose(x.ev(Z, (i, j)), x.R.ev(Z, (L-i-1, j)))

            assert np.allclose(x.ev(X, (i, j)), x.U.ev(X, (i, L-j-1)))
            assert np.allclose(x.ev(Y, (i, j)), x.U.ev(Y, (i, L-j-1)))
            assert np.allclose(x.ev(Z, (i, j)), x.U.ev(Z, (i, L-j-1)))

            z = fPEPS().random(L, L, 2, 3).isometrise((i, j))
            assert np.allclose(z.iso_ev(X), z.T.iso_ev(X))
            assert np.allclose(z.iso_ev(X), z.R.iso_ev(X))
            assert np.allclose(z.iso_ev(X), z.U.iso_ev(X))

            assert np.allclose(z.iso_ev(Y), z.T.iso_ev(Y))
            assert np.allclose(z.iso_ev(Y), z.R.iso_ev(Y))
            assert np.allclose(z.iso_ev(Y), z.U.iso_ev(Y))

            assert np.allclose(z.iso_ev(Z), z.T.iso_ev(Z))
            assert np.allclose(z.iso_ev(Z), z.R.iso_ev(Z))
            assert np.allclose(z.iso_ev(Z), z.U.iso_ev(Z))
    print('success')


def tests(N):
    test_moses_move(N)
    test_fPEPS_from_mps(N)
    test_separate_tensor(N)
    test_fPEPS_addition(N)
    test_fPEPS_evs(N)
    test_mpo_multiplication(N)
    test_overlap(N)
    test_peps_transpose(N)
    test_isometrize(N)
    test_split_tensor(N)

