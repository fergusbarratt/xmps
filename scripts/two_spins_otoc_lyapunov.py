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
mps = fMPS().left_from_state(tens_0_2).left_canonicalise(1)
F = Trajectory(mps)

F.H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2]
F.run_name = 'two_spins_int'
exps1, lys1 = F.lyapunov(T, m=1)
F.save()

F.H = [Sx1@Sx2+Sy1@Sy2+Sz1@Sz2+Sx1-Sz2]
F.run_name = 'two_spins_chaos'
exps2, lys2 = F.lyapunov(T, m=1)
F.save()

mps = fMPS().left_from_state(tens_0_2)
Ws1 = Trajectory(mps, Sx1@Sx2+Sy1@Sy2+Sz1@Sz2, fullH=True).ed_OTOC(T_, (Sz1, Sz1))

mps = fMPS().left_from_state(tens_0_2)
Ws2 = Trajectory(mps, F.H[0], fullH=True).ed_OTOC(T_, (Sz1, Sz1))

fig, ax = plt.subplots(2, 2, sharex=True)
ax[0][0].plot(T, exps1)
ax[0][0].set_title('D=1 no chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2$')

ax[1][0].plot(T, exps2)
ax[1][0].set_title('D=1 chaos: $H=S_x^1S_x^2+S_y^1S_y^2+S_z^1S_z^2 +S_x^1-S_z^2$')

ax[0][1].plot(T_, Ws1)
ax[0][1].set_title('otoc no chaos')

ax[1][1].plot(T_, Ws2)
ax[1][1].set_title('otoc chaos')
plt.savefig('images/exps_.pdf')
fig.tight_layout()
plt.show()
