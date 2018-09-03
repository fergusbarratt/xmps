'''various tests for MPS'''
from numpy import trace, dot, allclose, identity, array, diag, count_nonzero
from numpy import isclose, tensordot, squeeze, inf, sum, abs
from numpy.linalg import norm
from tensor import H


def clip(mats, tol):
    """clip: set elements below a tolerance to zero

    :param mats: matrices
    :param tol: tolerance
    """
    ret = []
    for mat in mats:
        a = mat.shape
        mat[abs(mat)<tol] = 0
        assert mat.shape == a
        ret.append(mat)
    return ret

def is_right_canonical(mats, c=1e-8, z=1e-8, error=False):
    """is_right_canonical: test is a set of arrays is right canonical
    (this is the wrong usage of canonical: only sum AAt = I)

    :param mats: matrices to test
    :param c: error tolerance
    :param z: clip to zero below z
    :error: print error information
    """
    Is = [tensordot(x, H(x), [[0, -1], [0, 1]]) for x in mats]
    Is = clip(Is, z)
    zero = [identity(max(x.shape))-x for x in Is]
    zero = clip(zero, z)
    close = [sum(z) for z in zero]
    if error:
        return allclose(close, 0, atol=c), close
    return allclose(close, 0, atol=c)

def is_left_canonical(mats, c=1e-8, z=1e-8, error=False):
    """is_left_canonical  test is a set of arrays is left canonical
    (this is the wrong usage of canonical: only sum AtA = I)
    WHY DOES TOLERANCE NEED TO BE SO HIGH TO PASS

    :param mats: matrices to test
    :param c: error tolerance
    :param z: clip to zero below z
    :error: print error information
    """
    Is = [tensordot(H(x), x, [[0, -1], [0, 1]]) for x in mats]
    Is = clip(Is, z)
    zero = [identity(max(x.shape))-x for x in Is]
    zero = clip(zero, z)
    close = [sum(z) for z in zero]
    # implications of sum/norm here - maybe averaging out error is ok
    if error:
        return allclose(close, 0, atol=c), close
    return allclose(close, 0, atol=c)

def is_right_env_canonical(mats, sings):
    """is_right_env_canonical: test the other half of canonical condition
    sum At L_n_1 A = L_n

    :param mats:
    :param sings:
    """
    Ls = [tensordot(dot(H(A), L), A, [[0, -1], [0, 1]])
            for (A, L) in zip(mats, sings)]
    close = array([allclose(l, x) for l, x in zip(Ls[:-1], sings[1:])])
    return close.all()

def is_left_env_canonical(mats, sings):
    """is_left_env_canonical: test the other half of canonical condition
    sum B_n L_n+1 B_nt = L_n
    """
    Ls = [tensordot(dot(A, L), H(A), [[0, -1], [0, 1]])
            for (A, L) in zip(mats, sings[1:])]
    close = array([allclose(l, x) for l, x in zip(Ls[:-1], sings[:-1])])
    return close.all()

def is_full_rank(sings):
    """is_full_rank: True if a list of singular matrices has no zero diagonals

    :param sings: list of singular matrices
    """
    sings = [diag(S) for S in sings]
    return sum([S.size - count_nonzero(S) for S in sings]) == 0

def has_trace_1(sings, error=False):
    """has_trace_1: True if a list of singular matrices have trace 1

    :param sings: singular matrices
    """
    er = array([trace(L) - 1.0 for L in sings])
    if error:
        return allclose(er, 0), er
    return allclose(er, 0)

