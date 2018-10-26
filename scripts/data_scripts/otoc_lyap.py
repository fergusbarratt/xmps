from sat import λ, λ_, D
import matplotlib.pyplot as plt
from numpy import linspace, load, cumsum as cs, arange as ar, exp
from numpy import log, array
def av(lys):
    return cs(lys)/ar(1, len(lys)+1)
def rm(lys):
    a = [lys[0]]
    for i in lys:
        a.append(i if i>a[-1] else a[-1])
    return a

St = load('data/S.npy')
y = load('data/otocs.npy')
Ds = rm(D(St))

dt = 0.15 
z = 2*dt*cs(λ(Ds))
z_ = dt*cs(λ_(Ds))

k = 500
plt.plot(linspace(0, 100, 2000)[:k], y[:k]-y[k], label='$log(OTOC(Sz^3, Sz^3)(t))$')
#plt.plot(linspace(0, 100, 1999), av(y[1:]), label='$av(log(OTOC(Sz^3, Sz^3)(t)))$')
plt.plot(linspace(0, 100, 1999)[:k], z[:k]-z[k], label='$\int^t\lambda_m(D(t))dt$')
#plt.plot(linspace(0, 100, 1999), z_, label='$\int^t\lambda_{ks}(D(t))dt$')
#plt.plot(linspace(0, 100, 1999), av(log(z)-log(z)[0]), label='$av(log(N\int^t\lambda(D(t))dt))$')
#plt.plot(linspace(0, 100, 1999), x, label='$S_E(t)$')
plt.title('time averages')
plt.legend()
#plt.savefig('images/time_averages.pdf')
plt.show()
