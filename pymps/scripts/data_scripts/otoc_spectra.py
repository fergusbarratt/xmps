from numpy import load, array
import matplotlib.pyplot as plt

T = linspace(0, 100, 2000)

x = array(list(map(max, load('data/exps.npy'))))
y = load('data/otocs.npy')
z = load('data/S.npy')

plt.scatter([1, 2, 3, 4, 5, 6, 7, 8], x, marker='x')
plt.plot(T, z)
plt.plot(T, y)

plt.show()
