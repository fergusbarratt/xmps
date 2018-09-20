import unittest
from fMPS import fMPS
from iMPS import iMPS
from numpy import zeros, array, isclose, allclose, pad, sqrt, stack, identity
from tensor import H
from spin import spins, ladders

Sx, Sy, Sz = spins(0.5)

def bell(i):
    """bell
    i=1: 11+00
    i=2: 11-00
    i=3: 10+01
    i=4: 10-01
    """
    matrices = [zeros((2, 1, 2))+0.j, zeros((2, 2, 1))+0.j]
    if i == 1:
        matrices[0][0] = array([0, 1])
        matrices[0][1] = array([1, 0])
        matrices[1][0] = array([[0], [1]])
        matrices[1][1] = array([[1], [0]])
        return fMPS(matrices, 2).left_canonicalise()
    elif i == 2:
        matrices[0][0] = array([0, 1])
        matrices[0][1] = array([1, 0])
        matrices[1][0] = array([[0], [1]])
        matrices[1][1] = -array([[1], [0]])
        return fMPS(matrices, 2).left_canonicalise()
    elif i == 3:
        matrices[0][0] = array([0, 1])
        matrices[0][1] = array([1, 0])
        matrices[1][0] = array([[1], [0]])
        matrices[1][1] = array([[0], [1]])
        return fMPS(matrices, 2).left_canonicalise()
    elif i == 4:
        matrices[0][0] = array([0, 1])
        matrices[0][1] = array([1, 0])
        matrices[1][0] = array([[1], [0]])
        matrices[1][1] = -array([[0], [1]])
        return fMPS(matrices, 2).left_canonicalise()

def comp_z(i):
    """comp

    i=1: (0, 0)
    i=2: (0, 1)
    i=3: (1, 0)
    i=4: (1, 1)
    """
    matrices = [zeros((2, 1, 1)), zeros((2, 1, 1))]
    if i == 1:
        matrices[0][0] = 1
        matrices[1][0] = 1
    if i == 2:
        matrices[0][0] = 1
        matrices[1][1] = 1
    if i == 3:
        matrices[0][1] = 1
        matrices[1][0] = 1
    if i == 4:
        matrices[0][1] = 1
        matrices[1][1] = 1
    
    matrices[0] = pad(matrices[0], ((0, 0), (0, 0), (0, 1)), 'constant')
    matrices[1] = pad(matrices[1], ((0, 0), (0, 1), (0, 0)), 'constant')
    return fMPS(matrices, 2)

def comp_x(i):
    """comp_x
    
    i=1: (+, +)
    i=2: (-, +)
    i=3: (+, -)
    i=4: (-, -)
    """
    matrices = [zeros((2, 1, 1))+1j*zeros((2, 1, 1)), 
                zeros((2, 1, 1))+1j*zeros((2, 1, 1))]
    if i == 1:
        matrices[0][0] = 1
        matrices[0][1] = 1
        matrices[1][0] = 1
        matrices[1][1] = 1
        return fMPS(matrices, 2).left_canonicalise()
    if i == 2:
        matrices[0][0] = 1j  # a1a2 = 00 = -1, a1b2 = 01 = 1, b1a2 = 10 = -1, b1b2 = 11 = 1
        matrices[0][1] = 1j
        matrices[1][0] = 1j 
        matrices[1][1] = -1j
    if i == 3:
        matrices[0][0] = -1j
        matrices[0][1] = 1j
        matrices[1][0] = 1j  # a1a2 = 00 = 1, a1b2 = 01 = 1, b1a2 = 10 = -1, b1b2 = 11 = -1
        matrices[1][1] = 1j
    if i == 4:
        matrices[0][0] = 1  # a1a2 = 00 = 1, a1b2 = 01 = -1, b1a2 = 10 = -1, b1b2 = 11 = 1
        matrices[0][1] = -1
        matrices[1][0] = 1
        matrices[1][1] = -1

    matrices[0] = pad(matrices[0], ((0, 0), (0, 0), (0, 1)), 'constant')
    matrices[1] = pad(matrices[1], ((0, 0), (0, 1), (0, 0)), 'constant')
    return fMPS(matrices, 2).left_canonicalise()

def i_comp_z(i):
    """i_comp

    i=1: (0)
    i=2: (1)
    """
    matrices = [zeros((2, 1, 1))]
    if i == 1:
        matrices[0][0] = 1
        matrices[0][1] = 0
    if i == 2: 
        matrices[0][0] = 0
        matrices[0][1] = 1
    return iMPS(matrices) 

def AKLT():
    """AKLT"""
    sx, sy, sz = spins(0.5)
    sp, sm = ladders(0.5)
    
    Ap = sqrt(2/3)*sp
    A0 = -sqrt(1/2)*sz
    Am= -sqrt(2/3)*sm
    return iMPS([stack([Ap, A0, Am], 0)])

class TestExamples(unittest.TestCase):

    def test_i_comp_z(self):
        self.assertTrue(i_comp_z(1).E(Sz)==1)
        self.assertTrue(i_comp_z(2).E(Sz)==-1)
        self.assertTrue(i_comp_z(1).E(Sx)==0)
        self.assertTrue(i_comp_z(2).E(Sx)==0)
        self.assertTrue(i_comp_z(1).E(Sy)==0)
        self.assertTrue(i_comp_z(2).E(Sy)==0)

    def test_bell_sz(self):
        for n in range(1, 5):
            self.assertTrue(isclose(0, bell(n).E(Sz, 0)))
            self.assertTrue(isclose(0, bell(n).E(Sz, 1)))

    def test_bell_orthogonal(self):
        for n in range(1, 5):
            for m in range(1, 5):
                if n == m:
                     break
                self.assertTrue(isclose(bell(n).overlap(bell(m)), 0))

    def test_bell_norms(self):
        for n in range(1, 5):
            self.assertTrue(isclose(bell(n).norm(), 1))

    def test_comp_z_orthogonal(self):
        for n in range(1, 5):
            for m in range(1, 5):
                if n == m:
                     break
                self.assertTrue(isclose(comp_z(n).overlap(comp_z(m)), 0))

    def test_comp_z_norms(self):
        for n in range(1, 5):
            self.assertTrue(isclose(comp_z(n).norm(), 1))

    def test_comp_z_sz(self):
        es = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for m in range(1, 5):
            self.assertTrue(allclose((comp_z(m).E(Sz, 0), comp_z(m).E(Sz, 1)), es[m-1]))

    def test_comp_z_sx(self):
        for m in range(1, 5):
            self.assertTrue(allclose(comp_z(m).E(Sx, 0), comp_z(m).E(Sx, 1), 0))

    def test_comp_x_orthogonal(self):
        for n in range(1, 5):
            for m in range(1, 5):
                if n == m:
                     break
                self.assertTrue(isclose(comp_x(n).overlap(comp_x(m)), 0))

    def test_comp_x_norms(self):
        for n in range(1, 5):
            self.assertTrue(isclose(comp_x(n).norm(), 1))

    def test_comp_x_sx(self):
        es = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for m in range(1, 5):
            self.assertTrue(allclose((comp_x(m).E(Sx, 0), comp_x(m).E(Sx, 1)), es[m-1]))

    def test_comp_x_sz(self):
        for m in range(1, 5):
            self.assertTrue(allclose(comp_x(m).E(Sz, 0), comp_x(m).E(Sz, 1), 0))

if __name__ == '__main__':
    unittest.main(verbosity=1)
