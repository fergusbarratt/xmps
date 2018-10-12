import numpy as np
import matplotlib.pyplot as plt

dat = [np.load(str(D)+'.npy') for D in (1, 2, 3, 4)]
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
for m, x in enumerate(dat):
    ax[m].hist(x, bins=10)
plt.show()
