import unittest

from .tensor import unitary_extension, embed, H as cT, C as c, T, haar_unitary
from .tensor import deembed, eye_like

from numpy import transpose, prod, array, sum, sqrt, mean, real, imag, concatenate
from numpy import array, allclose, kron, tensordot, trace as tr, eye
from numpy.random import randn

from math import log as mlog
def log2(x): return mlog(x, 2)

from cirq import TwoQubitMatrixGate, LineQubit, H, S, measure, inverse as inv, Circuit
from cirq import CSWAP, X
from cirq.google import XmonSimulator

from scipy.linalg import norm
from scipy.optimize import minimize

def mat(v):
    '''helper function - put list of elements (real, imaginary) in a square matrix'''
    re, im = v[:4], v[4:]
    C = (re+im*1j).reshape(int(sqrt(len(v))), -1)
    return C

def demat(A):
    re, im = real(A).reshape(-1), imag(A).reshape(-1)  
    return concatenate([re, im], axis=0)

def to_unitaries_l(AL):
    Us = []
    for A in AL:
        d, D, _ = A.shape
        iso = A.transpose([1, 0, 2]).reshape(D*d, D)
        assert allclose(cT(iso)@iso, eye(2)) # left isometry
        U = unitary_extension(iso)
        assert allclose(U@cT(U), eye(4)) # unitary
        assert allclose(cT(U)@U, eye(4)) # unitary
        assert allclose(U[:iso.shape[0], :iso.shape[1]], iso) # with the isometry in it
        assert allclose(tensordot(U.reshape(2, 2, 2, 2), array([1, 0]), [2, 0]).reshape(4, 2), 
                        iso)

        #  ↑ j
        #  | |
        #  ---       
        #   u  = i--A--j
        #  ---      |
        #  | |      σ
        #  i σ 
        Us.append(U)

    return Us

def from_unitaries_l(Us):
    As = []
    for U in Us: 
        A = tensordot(U.reshape(*2*int(log2(U.shape[0]))*[2]), array([1, 0]), [2, 0]).transpose([1, 0, 2])
        As.append(A)
    return As
    
def get_env(Us, C0=randn(2, 2)+1j*randn(2, 2), reps=10000000, k=500):
    ''' return v satisfying

        | | |   | | | 
        | ---   | | |       
        |  v    | | |  
        | ---   | | |  
        | | |   | | |           (2)
        --- |   --- |  
         u  |    v  |  
        --- |   --- |  
        | | | = | | |             
        j | |   j | |  


        to precision k/100000
        '''
    X = []
    def obj(v0, UL=Us[0], reps=reps, X=X):
        C = mat(v0)
        r = 0
        qbs = [LineQubit(i) for i in range(3)]
        env_qbs = [LineQubit(i) for i in range(2)]
        sim = XmonSimulator()
        ###########################################################
        ###########################################################
        q, p = Circuit(), Circuit()
        U, V = TwoQubitMatrixGate(UL), TwoQubitMatrixGate(embed(C))
        q.append([V(qbs[1], qbs[2]), U(qbs[0], qbs[1])])
        p.append([V(qbs[0], qbs[1])])
        p.append([measure(qbs[0], key='q2')])
        q.append([measure(qbs[0], key='q2')])

        LHS = sim.run(q, repetitions=reps).measurements['q2']
        RHS = sim.run(p, repetitions=reps).measurements['q2']
        x = array(list(map(int, LHS))).mean()
        y = array(list(map(int, RHS))).mean()
        r+=(x-y)**2
        ###########################################################
        ###########################################################
        q, p = Circuit(), Circuit()
        p.append([V(qbs[0], qbs[1])])
        q.append([V(qbs[1], qbs[2]), U(qbs[0], qbs[1])])
        p.append([H(qb) for qb in qbs])
        q.append([H(qb) for qb in qbs])
        p.append([measure(qbs[0], key='q2')])
        q.append([measure(qbs[0], key='q2')])

        LHS = sim.run(q, repetitions=reps).measurements['q2']
        RHS = sim.run(p, repetitions=reps).measurements['q2']
        x = array(list(map(int, LHS))).mean()
        y = array(list(map(int, RHS))).mean()
        r+=(x-y)**2
        ###########################################################
        ###########################################################
        q, p = Circuit(), Circuit()
        p.append([V(qbs[0], qbs[1])])
        q.append([V(qbs[1], qbs[2]), U(qbs[0], qbs[1])])

        p.append([H(qb) for qb in qbs])
        p.append([inv(S(qb)) for qb in qbs])
        q.append([H(qb) for qb in qbs])
        q.append([inv(S(qb)) for qb in qbs])

        p.append([measure(qbs[0], key='q2')])
        q.append([measure(qbs[0], key='q2')])

        LHS = sim.run(q, repetitions=reps).measurements['q2']
        RHS = sim.run(p, repetitions=reps).measurements['q2']
        x = array(list(map(int, LHS))).mean()
        y = array(list(map(int, RHS))).mean()
        r+=(x-y)**2
        X.append((v0, sqrt(r)))
        if sqrt(r) < k/reps:
            print('s', sqrt(r), k/reps)
            raise ValueError
        else:
            print('f', sqrt(r), k/reps)
            del X[0]
        return r

    v0 = demat(C0)
    try:
        res = minimize(obj, v0, method='Powell')
        x = res.x
        prec = res.fun
    except ValueError:
        x, prec = X[0]
    V = embed(mat(x))
    return V

import cirq 

def get_env_x(Us, C0=randn(2, 2)+1j*randn(2, 2)):
    ''' return v satisfying

        | | |   | | | 
        | ---   | | |       
        |  v    | | |  
        | ---   | | |  
        | | |   | | |           (2)
        --- |   --- |  
         u  |    v  |  
        --- |   --- |  
        | | | = | | |             
        j | |   j | |  


        to precision k/100000
    '''
    U = Us[0]
    C = C0

    qbs = [LineQubit(i) for i in range(3)]
    env_qbs = [LineQubit(i) for i in range(2)]
    sim = cirq.Simulator()
    ###########################################################
    ###########################################################
    q, p = Circuit(), Circuit()
    U, V = TwoQubitMatrixGate(U), TwoQubitMatrixGate(embed(C))
    q.append([V(qbs[1], qbs[2]), U(qbs[0], qbs[1])])
    p.append([V(qbs[0], qbs[1])])
    x= sim.simulate(q)
    y= sim.simulate(p)
    print(x.final_simulator_state)
    return x


def to_circuit(A, env_prec=1e-4):
    """represent and optimize an iMPS on a quantum computer
    :param A: iMPS object: assume right canonical
    """ 
    qbs = [LineQubit(i) for i in range(3)]
    C = Circuit()

    Us = to_unitaries_r(A)
    ops = [TwoQubitMatrixGate(u) for u in Us] 
    # ops is a list of gates corresponding to the state. 
    V = get_env(ops, reps=1/env_prec)

    U = ops[0]
    V = TwoQubitMatrixGate(V)
    C.append([V(qbs[0], qbs[1]), U(qbs[1], qbs[2])])
    return C

def get_r(V):
    """get_r: return ↑ ↑
    | |
    ---
     v
    ___
    / |
    
    \ |
    ---
     v
    ___
    | |
    ↑ ↑
    """
    z = array([1, 0, 0, 0])
    v = (V@z).reshape(2, 2)
    return v@v.conj().T
