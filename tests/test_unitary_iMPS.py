import unittest

from pymps.iMPS import iMPS, Map, TransferMatrix
from pymps.tensor import embed, deembed
from pymps.unitary_iMPS import *

from numpy import allclose
from numpy.random import randn

class TestcMPS(unittest.TestCase):
    """TestcMPS"""

    def setUp(self):
        N = 50
        self.xs = [randn(2, 2)+1j*randn(2, 2) for _ in range(N)]
        self.As = [iMPS().random(2, 2).mixed() for _ in range(N)]

    def test_mat_demat(self):
        for x in self.xs:
            self.assertTrue(allclose(mat(demat(x)), x))

    def test_to_unitaries_l(self):
        for AL, AR, C in self.As:
            Us = to_unitaries_l(AL)
            As = from_unitaries_l(Us)
            self.assertTrue(allclose(As[0], AL[0]))

    def test_get_env(self):
        for AL, AR, C in self.As:
            UL = to_unitaries_l(AL)[0]
            x = get_env_x([UL])
            print('hello', type(x))

            raise Exception
            UL = to_unitaries_l(AL)[0]

            AL, AR = AL.data[0], AR.data[0]
            self.assertTrue(allclose(AL@C, C@AR))
            
            r = C@C.conj().T
            l = C.conj().T@C
            I = eye_like(l)

            self.assertTrue(Map(AL, AL).is_right_eigenvector(r))
            self.assertTrue(Map(AL, AL).is_left_eigenvector(I))

            self.assertTrue(Map(AR, AR).is_right_eigenvector(I))
            self.assertTrue(Map(AR, AR).is_left_eigenvector(l))

            VI = kron(embed(C), I) # VI
            IV = kron(I, embed(C)) # IV
            UI = kron(UL, I)       # UI
            IU = kron(I, UL)       # IU

            z = array([1, 0, 0, 0, 0, 0, 0, 0])
            Q1 = (UI@IV@z).reshape(2, 2, 2)
            E_left = tensordot(Q1.conj(), Q1, [[1, 2], [1, 2]]).T # why the transposes/conjugates?

            Q3 = (IV@z).reshape(2, 2, 2)
            E= tensordot(Q3.conj(), Q3, [[0, 2], [0, 2]]).T # why the transposes/conjugates?

            self.assertTrue(allclose(E_left, E))
            self.assertTrue(allclose(E, C@C.conj().T))
            v0 = demat(C)

            V = get_env([UL], reps=100000)
            C_ = (V@array([1, 0, 0, 0])).reshape(2, 2)
            from numpy.linalg import eigvals
            from scipy.linalg import norm
            def evals(A):
                e = eigvals(A)
                return sorted(e/norm(e))
            print(evals(C_@C_.conj().T))
            print(evals(C@C.conj().T))
            print(evals(C_.conj().T@C_))
            print(evals(C.conj().T@C))
            raise Exception

if __name__=='__main__':
    unittest.main(verbosity=2)
