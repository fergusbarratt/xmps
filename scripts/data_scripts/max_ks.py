from numpy import load, save, array
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
k = 10000
def av(lys):
    return (1/(dt))*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = av(abs(load('data/spectra/lyapunovs_L8_D1_N20800.npy'))[-k:])[-1]
#B = abs(load('data/spectra/lyapunovs_L8_D2_N20400.npy')[-1])
#C = abs(load('data/spectra/lyapunovs_L8_D3_N20400.npy')[-1])
#D = abs(load('data/spectra/lyapunovs_L8_D4_N20400.npy')[-1])
#E = abs(load('data/spectra/lyapunovs_L8_D5_N20400.npy')[-1])
#F = abs(load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1])
#G = abs(load('data/spectra/lyapunovs_L8_D7_N20400.npy')[-1])
#H = abs(load('data/spectra/lyapunovs_L8_D8_N21000.npy')[-1])
#I = abs(load('data/spectra/lyapunovs_L8_D9_N21000.npy')[-1])
#J = abs(load('data/spectra/lyapunovs_L8_D10_N21000.npy')[-1])
#K = abs(load('data/spectra/lyapunovs_L8_D11_N21000.npy')[-1])
#L = abs(load('data/spectra/lyapunovs_L8_D12_N21600.npy')[-1])

data = [A]#, B, C, D, E, F, G, H, I, J, K, L]
#save('data/spectra/maxs.npy', array(list(map(max, data))))
Ds = [1]#, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].scatter(Ds, list(map(sum, data)), marker='x')
ax[0].set_title('KS', loc='right')
ax[1].scatter(Ds, list(map(max, data)), marker='x')
ax[1].set_title('max', loc='right')
#plt.savefig('images/spectra/max_ks.pdf', bbox_inches='tight')
plt.show()
