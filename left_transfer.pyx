import numpy as np
cimport numpy as np
def lt(np.ndarray op, list As, int j, int i):
    Ls = [op]
    for m in range(i-1, j-1, -1):
        W = np.tensordot(As[m], np.tensordot(As[m].conj(), Ls[-1], [2, 1]), [[0, 2], [0, 2]])
        Ls.append(W)
    return Ls[::-1]
