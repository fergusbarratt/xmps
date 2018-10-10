from numpy import load, cumsum as cs, expand_dims as ed, arange as ar, array
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')

k = 15000
step = 100 
def av(lys, dt):
    return (1/(dt))*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)
A = load('data/spectra/int/lyapunovs_L8_D2_N2000.npy')[-k:][::step]
avA = av(A, 5e-2)
B = load('data/spectra/lyapunovs_L8_D2_N2000.npy')[-k:][::step]
avB = av(B, 5e-2)

fig, ax = plt.subplots(2, 1)
ax[0].hist(avA[-1], bins=10)
ax[1].hist(avB[-1], bins=10)
plt.show()
