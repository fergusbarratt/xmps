import numpy as np
from .ncon import ncon
from scipy.linalg import norm

def group_legs(T, legs):
    """ Function to group legs of a tensor together.
    Parameters
    ----------
    T: np.array tensor.
    legs: List of legs to group together. [[l1,l2,l3],[l4,l5]] corresponds to 
    grouping legs l1,l2,l3 together and l4,l5 together.
    Returns
    ----------
    T_: The grouped tensor
    pipe: A tuple of a permutation and the original leg shape. This can be passed
    to ungroup legs to put the legs back in the original form.
    """
    perm = []
    for leg in legs:
        perm.extend(leg)
    T_ = np.transpose(T, perm)
    m = 0
    new_shape = []
    old_shape = []
    for leg in legs:
        n = len(leg)
        new_shape.append(np.prod(T_.shape[m:m+n]))
        old_shape.append(T_.shape[m:m+n])
        m += n
    pipe = (perm, old_shape)
    T_ = T_.reshape(new_shape)
    return(T_, pipe)

def inverse_transpose(perm):
    """ Returns the inverse of a permutation """
    inv = [0]* len(perm)
    for i in range(len(perm)):
        inv[perm[i]] = int(i)
    return(inv)

def ungroup_legs(T, pipe):
    """ Ungroups the legs.
    Parameters
    ----------
    T: The tensor to ungroup
    pipe: A tuple where the first element is a permutation and the second is 
    the original shape. These are the outputs of group_legs
    Returns
    ----------
    T_: The original tensor
    """
    perm, old_shape = pipe
    if (len(old_shape) != T.ndim):
        raise ValueError("Dimensions of shape and tensor must match")
    shape = []
    
    for i in range(len(old_shape)):
        if len(old_shape[i]) == 1:
            shape.append(T.shape[i])
        else: 
            shape.extend(old_shape[i])
    
    T_ = np.reshape(T, shape)
    T_ = T_.transpose(inverse_transpose(perm))
    return(T_)

def E2(U, θ):
    '''U has shape (d1, d2, d1, d2), θ has shape (d1, d2, Xl, Xr)'''
    Uθ = np.tensordot(U, θ , [[2, 3], [0, 1]])
    UE2 = ncon([U.conj(), Uθ, Uθ.conj(), Uθ, Uθ.conj()], [[7, 8, -1, -2], [7, 8, 1, 2], [3, -4, 4, 2], [3, 5, 4, 6], [-3, 5, 1, 6]])
    return UE2

def renyi(U, θ):
    d1, d2, Xl, Xr = θ.shape
    return -np.real(np.log(np.trace(ncon([U, E2(U, θ)], [[-1, -2, 1, 2], [1, 2, -3, -4]]).reshape(d1*d2, d1*d2))))

def U2(*args, **kwargs):
    pass

#U_ = np.linalg.qr(np.random.randn(4, 4)+1j*np.random.randn(4, 4))[0]
#U = U_.reshape(2, 2, 2, 2)
#psi = np.random.randn(2, 2, 5, 5,)+0j
#psi+=np.random.randn(*psi.shape)*1j
#psi /= norm(psi)
#
#print(renyi(U, psi))
#
#U = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])).reshape(2, 2, 2, 2)
#psi = np.kron(np.array([1, 0]), np.array([1, 0]))
#psi = psi.reshape(2, 2, 1, 1)
#print(renyi(U, psi))
#raise Exception
from scipy.linalg import polar

def renyi_2_disentangler(psi, eps=1e-6, max_iter=1000):
    Ss = []
    d1, d2, Xl, Xr = psi.shape
    U = np.eye(d1 * d2, dtype = psi.dtype)
    #H = np.random.randn(d1*d2, d1*d2)+1j*np.random.randn(d1*d2, d1*d2)
    #H = H+H.conj().T
    #U = np.linalg.qr(H)[0]
    m = 0
    go = True
    stop=False
    while not stop:
        while m < max_iter and go:
            Unew, _ = polar(E2(U.reshape(d1, d2, d1, d2), psi).reshape(d1*d2, d1*d2))
            Unew, _ = polar(E2(Unew.reshape(d1, d2, d1, d2), psi).reshape(d1*d2, d1*d2)) # with only one iteration S oscillates. v. weird

            S = renyi(Unew.reshape(d1, d2, d1, d2), psi)
            Ss.append(S)
            U = Unew

            if m > 1:
                print(np.abs(Ss[-2]-Ss[-1]))
                go = np.abs(Ss[-2] - Ss[-1]) > eps
            m += 1
        if go:
            H = np.random.randn(d1*d2, d1*d2)+1j*np.random.randn(d1*d2, d1*d2)
            H = H+H.conj().T
            U = np.linalg.qr(H)[0]
            m = 0
        else:
            stop=True

    return U, Ss

#X = 3
#d = 2
#psi = np.random.randn(d, d, X, X)+0j
#psi+=np.random.randn(*psi.shape)*1j
#psi /= norm(psi)
#U, Ss = renyi_2_disentangler(psi)
#I = np.eye(d**2).reshape(d, d, d, d)
#
#print(np.exp(-renyi(U.reshape(d, d, d, d), psi)), np.exp(-renyi(I, psi)))
#import matplotlib.pyplot as plt
#plt.plot(Ss)
#plt.savefig('x.pdf', bbox_inches='tight')

