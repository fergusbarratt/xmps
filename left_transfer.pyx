from ncon import ncon
def lt(op, As, j, i):
    Ls = [op]
    oplinks = [2, 3]+list(range(-3, -len(op.shape)-1, -1))
    for m in reversed(range(j, i)):
        Ls.insert(0, ncon([As[m].conj(), As[m], Ls[0]], [[1, -2, 3], [1, -1, 2], oplinks], [2, 3, 1]))
    return Ls
