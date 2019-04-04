from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
G = abs(load('data/spectra/lyapunovs_L8_D9_N21000.npy')[-1])
print(sum(G))
print(max(G))
