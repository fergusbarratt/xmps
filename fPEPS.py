from itertools import product
from numpy.random import rand
from numpy import array, identity, tensordot, transpose, reshape
from numpy import allclose, prod
from tensor import structure, C
from copy import deepcopy
from qmb import sigmaz
from fMPS import fMPS
sz = sigmaz().full()

class fPEPS(object):
    """fPEPS: lists of lists of arrays with 5 indices"""
    def __init__(self, data=None):
        if data is not None:
            self.L1 = len(self.data)
            self.L2 = len(self.data[0])
            self.d = self.data[0][0].shape[0]
            self.data = data
        
    def create_structure(self, L, d, D):
        """create_structure

        :param L: Length -> square with sides L
        :param d: local state space dimension
        :param D: bond dimension
        generate the structure of a OBC fMPS
        """

        if D is None:
            D = d**L
        left = [(min(d**n, D), min(d**(n+1), D)) for n in range(L//2)]

        def transpose(ltups):
            l = []
            for tup in ltups:
                l.append((tup[1], tup[0]))
            return l

        if L % 2 == 0:
            structure = left + transpose(left[::-1])
        else:
            structure = left + [(left[-1][-1], transpose(left[::-1])[0][0])] +\
                        transpose(left[::-1])

        lr = [structure for _ in range(L)]
        ud = [[structure[i]]*L for i in range(L)]

        return [[(d, *lr[i][j], *ud[i][j]) for i in range(L)] for j in range(L)]

    def random(self, L, d, D):
        self.L = L
        self.d = d
        self.D = D
        shapes = self.create_structure(L, d, D)
        self.data = [[rand(*shapes[i][j])+1j*rand(*shapes[i][j]) for i in range(L)] for j in range(L)]
        return self

    def structure(self):
        L = self.L
        return [[self.data[i][j].shape for i in range(L)] for j in range(L)]

    def mul(self, op, site):
        i, j = site
        self.data[i][j] = tensordot(op, self.data[i][j], [1, 0])
        return self
    
    def E(self, op, site, chi=1000):
        A = deepcopy(self.data)
        B = self.mul(op, site).data
        i, j = site
        L = self.L
        flat =  [[reshape(transpose(tensordot(A[i][j], C(B[i][j]), [0, 0]), 
                                    [0, 4, 1, 5, 2, 6, 3, 7]),
                          [X**2 for X in A[i][j].shape[1:]])
                  for i in range(self.L)]
                 for j in range(self.L)]


        for i in range(1, L):
            flat[i] = [transpose(tensordot(Mi, Mi_1, [1, 0]),
                                   [0, 3, 1, 2, 4, 5]).reshape(Mi.shape[0], 
                                                               Mi_1.shape[1], 
                                                               Mi.shape[2]*Mi_1.shape[2], 
                                                               Mi.shape[3]*Mi_1.shape[3])
                         for Mi, Mi_1 in zip(flat[i-1], flat[i])]
            shape = flat[i][0].shape[:2]
            ## convert the row to an MPS, compress, return 
            #flat[i] = [x.reshape(*shape, *x.shape[1:])
            #           for x in fMPS([f.reshape(prod(f.shape[:2]), f.shape[2], f.shape[3]) for f in flat[i]], 
            #                         d=prod(shape)).left_canonicalise(chi).data]
        line = flat[-1]
        for i in range(1, L):
            line[i] = line[i-1] @ line[i] 

        return line[-1][0][0][0][0]

    def norm(self):
        site = 0, 0
        op = identity(self.d)
        return self.E(op, site)

f = fPEPS().random(3, 2, 2)
print(f.E(sz, [1, 1])/f.norm())
