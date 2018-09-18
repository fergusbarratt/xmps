from numpy import load
from seaborn import distplot
import matplotlib.pyplot as plt

exps_i1 = load('../data/i_lyapunovs_L6_D1.npy')
exps_c1 = load('../data/c_lyapunovs_L6_D1.npy')
exps_i2 = load('../data/i_lyapunovs_L6_D2.npy')
exps_c2 = load('../data/c_lyapunovs_L6_D2.npy')
exps_i3 = load('../data/i_lyapunovs_L6_D3.npy')
exps_c3 = load('../data/c_lyapunovs_L6_D3.npy')

fig, ax = plt.subplots(6, 2, sharex='col', sharey='col')
ax[0][0].plot(exps_i1)
distplot(exps_i1[-1], ax=ax[0][1], bins=6)
ax[1][0].plot(exps_c1)
distplot(exps_c1[-1], ax=ax[1][1], bins=6)
ax[2][0].plot(exps_i2)
distplot(exps_i2[-1], ax=ax[2][1], bins=6)
ax[3][0].plot(exps_c2)
distplot(exps_c2[-1], ax=ax[3][1], bins=6)
ax[4][0].plot(exps_i3)
distplot(exps_i3[-1], ax=ax[4][1], bins=6)
ax[5][0].plot(exps_c3)
distplot(exps_c3[-1], ax=ax[5][1], bins=6)
