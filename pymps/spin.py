import numpy as np
from numpy import array, allclose, sqrt, zeros, reshape
from numpy import block, eye, trace
from numpy import tensordot, kron, identity, diag, arange, allclose
from scipy.sparse.linalg import expm
from itertools import product
from functools import reduce
from math import log as logd, sqrt

def is_swap_symmetric(U):
    return allclose(U-swap_antisymmetrise(U), 0)

def swap_symmetrise(U):
    return (U+swap()@U@swap())/2

def swap_antisymmetrise(U):
    return (U+swap()@U@swap())/2

def swap():
    return array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0], 
                  [0, 0, 0, 1]])

def levi_civita(dim):
    """levi_civita symbol rank dim
    https://bitbucket.org/snippets/lunaticjudd/doqp7/python-implementation-of-levi-civita

    :param dim:
    """
    def perm_parity(a,b):
        """Modified from
        http://code.activestate.com/recipes/578236-alternatve-generation-of-the-parity-or-sign-of-a-p/"""
        
        a = list(a)
        b = list(b)

        if sorted(a) != sorted(b): return 0
        inversions = 0
        while a:
            first = a.pop(0)
            inversions += b.index(first)
            b.remove(first)
        return -1 if inversions % 2 else 1

    def loop_recursive(dim,n,q,s,paritycheck):
        if n < dim:
            for x in range(dim):
                q[n] = x
                loop_recursive(dim,n+1,q,s,paritycheck)
        else:
            s.append(perm_parity(q,paritycheck))
    qinit = zeros(dim)
    paritycheck = range(dim)
    flattened_tensor = []
    loop_recursive(dim,0,qinit,flattened_tensor,paritycheck)

    return reshape(flattened_tensor,[dim]*dim)

def tensor(ops):
    return reduce(kron, ops)

def n_body(op, i, n, d=None):
    """n_body: n_body versions of local operator

    :param op: operator to tensor into chain of identities
    :param i: site for operator. 1-indexed
    :param n: length of chain
    :param d: local dimension of identities to tensor. If None, use size of op
    """
    i = i-1
    if d is None:
        d = op.shape[0]
        l = [identity(d)*(1-m) + op*m for m in map(lambda j: int(not i-j), range(n))]
    else:
        l = [identity(d) for _ in range(n+1-int(logd(op.shape[0], d)))]
        l.insert(i+1, op)
        #l = [op if j==i else identity(d) for j in range(n-int(logd(op.shape[0], d)))]
    if not i < n:
        raise Exception("i must be less than n")
    return tensor(l)

def spins(S):
    """spins. returns [Sx, Sy, Sz] for spin S

    :param S: spin - must be in [0.5, 1, 1.5]
    """
    def spin(S, i):
        """i=0: Sx
           i=1: Sy
           i=2: Sz
           """
        if S == 1/2:
            if i == 0:
                return 1/2*array([[0, 1], 
                                  [1, 0]])
            if i == 1: 
                return 1/2j*array([[0  , 1]   ,
                                   [-1 , 0]])
            if i == 2:
                return 1/2*array([[1 , 0 ] ,
                                  [0 , -1]] )
        if S == 1:
            if i == 0:
                return 1/sqrt(2)*array([[0, 1, 0],
                                        [1, 0, 1], 
                                        [0, 1, 0]])
            if i == 1:
                return -1j/sqrt(2)*array([[0 , 1 , 0],
                                          [-1, 0 , 1],
                                          [0 , -1, 0]])
            if i == 2:
                return array([[1, 0, 0 ], 
                              [0, 0, 0 ], 
                              [0, 0, -1]])
        if S == 3/2:
            if i == 0:
                return 1/2*array([[0       , sqrt(3) , 0       , 0      ],
                                  [sqrt(3) , 0       , 2       , 0      ],
                                  [0       , 2       , 0       , sqrt(3)] ,
                                  [0       , 0       , sqrt(3) , 0      ]])
            if i == 1:
                return 1/2j*array([[0        , sqrt(3) , 0        , 0       ],
                                   [-sqrt(3) , 0       , 2        , 0       ],
                                   [0        , -2      , 0        , sqrt(3) ],
                                   [0        , 0       , -sqrt(3) , 0       ]])
            if i == 2:
                return array([[3/2 , 0   , 0    , 0   ],
                              [0   , 1/2 , 0    , 0   ],
                              [0   , 0   , -1/2 , 0   ],
                              [0   , 0   , 0    , -3/2]])

    def arc(x):
        return array(list(x))

    def Cp(j, m): return sqrt((j-m)*(j+m+1))
    def Cm(j, m): return sqrt((j+m)*(j-m+1))
    def Sp(j): return diag(arc(Cp(j, m) for m in arange(j-1, -j-1, -1)), 1)
    def Sm(j): return diag(arc(Cm(j, m) for m in arange(j, -j, -1)), -1)

    def Sx(j): return (Sp(j)+Sm(j))/2
    def Sy(j): return (Sp(j)-Sm(j))/2j
    def Sz(j): return diag(arc(arange(j, -j-1, -1)))

    return (Sx(S), Sy(S), Sz(S))

def ladders(S):
    """ladders

    :param S: spin
    returns: list: [S_-, S_+]
    """
    def ladder(S, pm):
        """ladder: return S_+ and S_- for given S

        :param S: spin
        """
        if S == 1/2:
            if pm == 1:
                return array([[0, 1], [0, 0]])
            if pm == -1:
                return array([[0, 0], [1, 0]])
        if S == 1:
            if pm == 1:
                return sqrt(2)*array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
            if pm == -1:
                return sqrt(2)*array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        if S == 3/2:
            if pm == 1:
                return array([[0 , sqrt(3) , 0 , 0      ],
                              [0 , 0       , 2 , 0      ],
                              [0 , 0       , 0 , sqrt(3)],
                              [0 , 0       , 0 , 0      ]])
            if pm == -1:
                return array([[0       , 0 , 0       , 0],
                              [sqrt(3) , 0 , 0       , 0],
                              [0       , 2 , 0       , 0],
                              [0       , 0 , sqrt(3) , 0]])
    return [ladder(S, pm) for pm in [-1, 1]]

def paulis(S):
    Sx, Sy, Sz = spins(0.5)
    return 2*Sx, 2*Sy, 2*Sz

def N_body_spins(S, i, N):
    """N_body_spiNs: S_x^i etc. -> local spiN operators with ideNtities 
       teNsored iN oN either side

    :param S: spiN
    :param i: site for local spiN operator: 1-iNdexed
    :param N: leNgth of chaiN 
    """
    return [n_body(s, i, N) for s in spins(S)]

def N_body_paulis(S, i, N):
    """N_body_spiNs: S_x^i etc. -> local pauli operators with ideNtities 
       teNsored iN oN either side

    :param S: spiN
    :param i: site for local spiN operator: 1-iNdexed
    :param N: leNgth of chaiN 
    """
    return [n_body(s, i, N) for s in paulis(S)]

def N_body_ladders(S, i, N):
    """N_body_ladders: S_+^i etc. -> local spiN ladder operators 
       with ideNtities teNsored iN oN either side

    :param S: spiN
    :param i: site for local spiN operator: 1-iNdexed
    :param N: leNgth of chaiN 
    """
    return [n_body(s, i, N) for s in ladders(S)]

def swap_rows(A, i, j):
    B = A.copy()
    B[[i, j]] = B[[j, i]]
    return B

def comm(A, B):
    return A@B - B@A

def acomm(A, B):
    return A@B + B@A

def CR(Sx, Sy, Sz):
    """CR: Determine if a set of spin operators satisfy spin commutation relations
    """
    S = [Sx, Sy, Sz]
    satisfied = True 
    eps = levi_civita(3)
    for j, k in product(range(3), range(3)):
        satisfied = satisfied and allclose(comm(S[j], S[k]), 
                                           tensordot(eps[j, k]*1j, S, [0, 0]))
    return satisfied   

def pCR(Sx, Sy, Sz):
    """CR: Determine if a set of pauli operators satisfy pauli commutation relations
    """
    S = [Sx, Sy, Sz]
    satisfied = True 
    eps = levi_civita(3)
    for i, j in product(range(3), range(3)):
        satisfied = satisfied and allclose(comm(S[i], S[j]), 
                                           tensordot(2*eps[i, j, :]*1j, S, [0, 0]))
    return satisfied   

def spinHamiltonians(object):
    """1d spin Hamiltonians"""
    def __init__(self, S, finite=True):
        """__init__"""
        self.Sx = lambda i:   N_body_spins(S, i, 2)[0] 
        self.Sy = lambda i:   N_body_spins(S, i, 2)[1]
        self.Sz = lambda i:   N_body_spins(S, i, 2)[2] 
        self.finite = finite

    def nn_general(self, Jx, Jy, Jz, hx, hy, hz):
        """nn_general: nn spin model with all nn couplings (Jx, Jy, Jz)
           and fields (hx, hy, hz). All couplings positive by default
        """
        Sx, Sy, Sz = self.Sx, self.Sy, self.Sz
        h_bulk = Jx * Sx(1) @ Sx(2) + \
                 Jy * Sy(1) @ Sy(2) + \
                 Jz * Sz(1) @ Sz(2) + \
                 hx * Sx(1) + \
                 hy * Sy(1) + \
                 hz * Sz(1) + \
                 hx * Sx(2) + \
                 hy * Sy(2) + \
                 hz * Sz(2)

        if self.finite:
            N = self.N
            h_end = h_bulk + \
                     hx * Sx(2) + \
                     hy * Sy(2) + \
                     hz * Sz(2)

            return [h_bulk]*(N-2) + [h_end]
        else:
            return h_bulk

    def heisenberg_ferromagnet(self, J):
        """heisenberg_ferromagnet: -J \sum S_{i} S_{i+1}
        """
        return self.nn_general(-J, -J, -J, 0, 0, 0)

    def heisenberg_antiferromagnet(self, J):
        """heisenberg_antiferromagnet: J \sum S_{i} S_{i+1} 
        """
        return self.nn_general(J, J, J, 0, 0, 0)
    
    def XY(self, gamma):
        """XY: H = -\sum_{i} [(1+gamma) S_{i}^x S_{i+1}^x + (1-gamma) S_{i}^y S_{i+1}^y]
        """
        return self.nn_general(1+gamma, 1-gamma, 0, 0, 0, 0)

    def XXZ(self, J, delta):
        """XXZ: H = \sum_{i} J (S^x_{i} S^x_{i+1} + S^y_{i} S^y_{i+1}) + \Delta S^z_{i} S^z_{i+1}
        """
        return self.nn_general(J, J, delta, 0, 0, 0)

    def TFIM(self, l):
        """TFIM: H = -\sum_{i} [S_i^x + l S_{i}^z S_{i+1}^z]
        """
        return self.nn_general(0, 0, -l, -1, 0, 0)

    def AKLT(self):
        SS = self.heisenberg_antiferromagnet(1)
        if not self.finite:
            return 1/2 * SS + 1/6 * SS@SS  + 1/3
        else:
            N = self.N
            return [1/2*SS + 1/6 * SS@SS + 1/3]*(N-1)

assert all([CR(*spins(S)) for S in [0.5, 1, 1.5, 2., 2.5, 3]])
assert all([pCR(*paulis(S)) for S in [0.5, 1, 1.5]])

# From here on out it's Lie algebraish
def lambdas(S=0.5):
    """lambdas: generators of SU(4)
    :param S: spin
    """
    if S!=0.5:
        raise NotImplementedError
    else:
        Sx, Sy, Sz = paulis(0.5)
        O = zeros((2, 2))
        λ1 = block([[Sx, O], [O, O]])
        λ2 = block([[Sy, O], [O, O]])
        λ3 = block([[Sz, O], [O, O]])
        λ4 = array([[0, 0, 1, 0], 
                    [0, 0, 0, 0], 
                    [1, 0, 0, 0], 
                    [0, 0, 0, 0]])
        λ5 = array([[0, 0, -1j, 0], 
                    [0, 0, 0, 0], 
                    [1j, 0, 0, 0], 
                    [0, 0, 0, 0]])
        λ6 = array([[0, 0, 0, 0], 
                    [0, 0, 1, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, 0, 0]])
        λ7 = array([[0, 0, 0, 0], 
                    [0, 0, -1j, 0], 
                    [0, 1j, 0, 0], 
                    [0, 0, 0, 0]])
        λ8 = array([[1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, -2, 0], 
                    [0, 0, 0, 0]])*(1/sqrt(3))
        λ9 = array([[0, 0, 0, 1], 
                    [0, 0, 0, 0], 
                    [0, 0, 0, 0], 
                    [1, 0, 0, 0]])
        λ10 = array([[0, 0, 0, -1j], 
                     [0, 0, 0, 0], 
                     [0, 0, 0, 0], 
                     [1j, 0, 0, 0]])
        λ11 = array([[0, 0, 0, 0], 
                     [0, 0, 0, 1], 
                     [0, 0, 0, 0], 
                     [0, 1, 0, 0]])
        λ12 = array([[0, 0, 0, 0], 
                     [0, 0, 0, -1j], 
                     [0, 0, 0, 0], 
                     [0, 1j, 0, 0]])
        λ13 = array([[0, 0, 0, 0], 
                     [0, 0, 0, 0], 
                     [0, 0, 0, 1], 
                     [0, 0, 1, 0]])
        λ14 = array([[0, 0, 0, 0], 
                     [0, 0, 0, 0], 
                     [0, 0, 0, -1j], 
                     [0, 0, 1j, 0]])
        λ15 = array([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, -3]])*(1/sqrt(6))

        return array([λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8, λ9, λ10, λ11, λ12, λ13, λ14, λ15])

def slambdas(S=0.5):
    """slambdas: generators of swap symmetric subspace.
                 should prove this!

    :param S:
    """
    if S!=0.5:
        raise NotImplementedError
    else:
        Sx, Sy, Sz = paulis(0.5)
        I = eye(2)
        return array([kron(Sx, Sx), 
                      kron(Sy, Sy), 
                      kron(Sz, Sz), 
                      kron(Sx, Sy) + kron(Sy, Sx),
                      kron(Sx, Sz) + kron(Sz, Sx),
                      kron(Sy, Sz) + kron(Sz, Sy),
                      kron(Sx, I) + kron(I, Sx),  
                      kron(Sy, I) + kron(I, Sy),
                      kron(Sz, I) + kron(I, Sz)])
         
def antiferro():
    """Make an SU(2) spinor"""
    sp1, sm1 = N_body_ladders(0.5, 1, 2) # these are (sigmax+ isigmay) /2 = Sx+iSy
    sp2, sm2 = N_body_ladders(0.5, 2, 2)
    zp1, zm1 = (0.5*eye(4)+sz1), (0.5*eye(4)-sz1)
    zp2, zm2 = (0.5*eye(4)+sz2), (0.5*eye(4)-sz2)
    n_x = (sp1@sm2 + sm1@sp2)
    n_y = -1j*(sp1@sm2 - sm1@sp2)
    n_z = (sz2-sz1)/2
    return array([n_x, n_y, n_z])

def ferro():
    """Make another SU(2) spinor"""
    sp1, sm1 = N_body_ladders(0.5, 1, 2) # these are (sigmax+ isigmay) /2 = Sx+iSy
    sp2, sm2 = N_body_ladders(0.5, 2, 2)
    zp1, zm1 = (0.5*eye(4)+sz1), (0.5*eye(4)-sz1)
    zp2, zm2 = (0.5*eye(4)+sz2), (0.5*eye(4)-sz2)
    m_x = (sp1@sp2 + sm1@sm2)
    m_y = 1j*(sp1@sp2-sm1@sm2)
    m_z = (sz1+sz2)/2
    return array([m_x, m_y, m_z])

def locals():
    """local spinors"""
    Sx, Sy, Sz = spins(0.5)
    I = eye(2)
    return array([kron(I, Sx), kron(I, Sy), kron(I, Sz),
                  kron(Sx, I), kron(Sy, I), kron(Sz, I)])

def U1(v):
    """U1 local unitary
    """
    return expm(-1j*tensordot(v, locals[:3], [0, 0]))

def U2(v):
    """U2 local unitary
    """
    return expm(-1j*tensordot(v, locals[3:], [0, 0]))

def U4(v):
    """U4 two site unitary
    """
    return expm(-1j*tensordot(v, lambdas(), [0, 0]))

def U4s(v):
    """U4s two site symmetric unitary
    """
    return expm(-1j*tensordot(v, slambdas(), [0, 0]))

op = lambdas(0.5)
assert all([allclose(trace(o), 0) for o in op])
assert all([allclose(trace(o@o), 2) for o in op])

Sx1, Sy1, Sz1 = N_body_spins(1/2, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(1/2, 2, 2)

