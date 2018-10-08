import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins
from numpy import linspace
from numpy import load
import matplotlib.pyplot as plt

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

dt = 1e-2
t_fin = 10 
T = linspace(0, t_fin, int(t_fin//dt)+1)

tens_0_2 = load('fixtures/mat2x2.npy')
mps = fMPS().left_from_state(tens_0_2).left_canonicalise(1)
F = Trajectory(mps)

F.H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]
F.run_name = 'two_spins/int'
exps1, lys1 = F.lyapunov(T)
F.save()

F.H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2+Sx1-Sz2]
F.run_name = 'two_spins/chaos'
exps2, lys2 = F.lyapunov(T)
F.save()

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].plot(lys1)
ax[0].set_title('D=1 no chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2$')

ax[1].plot(lys2)
ax[1].set_title('D=1 chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2 +S_x^1-S_z^2$')

#plt.savefig('images/exps.pdf')
fig.tight_layout()
plt.show()
