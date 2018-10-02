from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
mpl.style.use('bmh')

F = load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1]
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
sns.distplot(F, fit=stats.semicircular, kde=False, rug=True,ax=ax)
fig.tight_layout()
ax.set_title('D=6, L=8', loc='right')
#plt.savefig('images/spectra/semicircle.pdf', bbox_inches='tight')
plt.show()
