import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins
from numpy import linspace, tensordot
from numpy import load, array
import matplotlib.pyplot as plt

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

dt = 1e-2
t_fin = 10
T = linspace(0, t_fin, int(t_fin//dt)+1)

tens_0_2 = load('fixtures/mat2x2.npy')
mps = fMPS().left_from_state(tens_0_2).left_canonicalise()
F = Trajectory(mps)

M = load('M.npy')
X1 = array([Sx1, Sy1, Sz1])
X2 = array([Sx2, Sy2, Sz2])

F.H = [tensordot(X1, tensordot(M, X2, [-1, 0]), [[0, -1], [0, 1]])]
F.run_name = 'two_spins/int'
X = F.lyapunov(T)
#F.stop()
#F.clear()

print(len(X))
plt.plot(X[0])
plt.show()
