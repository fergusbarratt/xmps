from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

F = load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1]
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 4))
print(len(F))
ax.hist(F, bins=8)
fig.tight_layout()
#plt.savefig('images/spectra/semicircle.pdf', bbox_inches='tight')
plt.show()
