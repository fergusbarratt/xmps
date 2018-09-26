import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fTDVP import Trajectory
from fMPS import fMPS
from numpy import linspace, load
import matplotlib.pyplot as plt
from spin import N_body_spins

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

dt = 1e-2
t_fin = 100 
T = linspace(0, t_fin, int(t_fin//dt)+1)
T_ = linspace(0, t_fin, 100)

tens_0_2 = load('fixtures/mat2x2.npy')
H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2+Sx1-Sz2]

mps = fMPS().left_from_state(tens_0_2)
Ws = Trajectory(mps, H[0], fullH=True).ed_OTOC(T_, (Sz1, Sz1))

mps = mps.left_canonicalise(1)
exps2, lys2 = Trajectory(mps, H).lyapunov(T, m=1)
H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]
exps1, lys1 = Trajectory(mps, H).lyapunov(T, m=1)



fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(T, exps1)
ax[0].set_title('D=1 no chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2$')

ax[1].plot(T, exps2)
ax[1].set_title('D=1 chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2 +S_x^1-S_z^2$')

ax[2].plot(T_, Ws)
ax[2].set_title('otoc')
#plt.savefig('images/exps.pdf')
fig.tight_layout()
plt.show()
