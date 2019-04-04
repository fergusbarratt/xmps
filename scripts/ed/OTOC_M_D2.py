import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir) 
from fMPS import fMPS
from tdvp.tdvp_fast import MPO_TFI
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array, cumsum as cs
from numpy import arange as ar, mean, eye, diag, max, exp, isnan, prod
from numpy.linalg import eigvalsh, svd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensor import H as cT, C as c
from tqdm import tqdm
from tdvp.ex import sympmat, williamson, tr_symp, trace_out
from scipy.sparse.linalg import expm, expm_multiply
Sx, Sy, Sz = spins(0.5)

L = 6

S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

ent = Sz1@Sz2
loc = Sx1+Sz1, Sx2+Sz2

listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
T = linspace(0, 2, 100)
D = 2 
A = fMPS().random(L, 2, D).left_canonicalise()
As_ = Trajectory(A,
                H=listH,
                W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).edint(T)

As = Trajectory(A,
                H=listH,
                W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).ts_int(T)

δt = T[1]-T[0]
Js = [A.jac(listH) for A in tqdm(As.mps_list())]

bosons_on_site = [prod(x) for x in As.mps_list()[-1].tangent_space_dims()]
total_bosons = cs(bosons_on_site)

Ms = As.Ms[1:]

n = Js[0].shape[0]//2
Ω = sympmat(n)
Cs = [(M@Ω)**2 for M in Ms]

def s_max(M):
    """return largest singular value
    """
    s_max = max(svd(M)[1])
    return s_max

def bosonic_renyi(M, js):
    nu = williamson(trace_out(M.T@M, js))[0]
    return sum(log(diag(nu)))

def lyapunov(M):
    """lyapunov: returns λ(t)
    """
    s_max = max(svd(M)[1])
    return log(s_max)/2

fig, ax = plt.subplots(L+1, 1, sharex=True, figsize=(7.5, 7.5))
for i, j in zip(range(L), range(L)):
    OTOC  = Trajectory(fMPS().load('fixtures/product{}.npy'.format(L)),
                       H=fullH,
                       W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)], fullH=True).ed_OTOC(T, (S_list[i][0], S_list[j][0]))
    for k in range(bosons_on_site[i]):
        print('plotting boson: ', total_bosons[i-1]+k)
        botoc = [C[total_bosons[i-1]+k, total_bosons[i-1]+k] for C in Cs]
        ax[i].plot(T, botoc, linewidth=1.3, label='bosons')
    #ax[i].legend(loc=1, bbox_to_anchor=(1.2, 1))
    ax[i].set_ylabel('site {}'.format(i+1))
    ax[i].plot(T, OTOC, linewidth=1.5, c='black', label='spins')
ax[-1].set_xlabel('t')

fig.suptitle('spin vs boson OTOC, $D={}$, Black lines exact spin OTOC $(Sx, Sx)$,\n coloured lines are bosonic OTOCs $(a+a^\dagger, a+a^\dagger)$'.format(D))
plt.tight_layout()
fig.subplots_adjust(top=0.92)

#ax[L].plot(T, As.renyi())
#ax[L].plot(T, As_.renyi())
ax[L].plot(T, [bosonic_renyi(M, range(total_bosons[L//2])) for M in Ms])

ax[L].plot(T, As.von_neumann()/D**2)
ax[L].plot(T, As_.von_neumann()/D**2)
ax[L].set_ylabel('$S_E$')
#plt.savefig('images/OTOC/D={}.pdf'.format(D), bbox_inches='tight')
plt.show()
