from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
G = abs(load('data/spectra/lyapunovs_L8_D8_N20400.npy')[-1])
print(sum(G))
print(max(G))
