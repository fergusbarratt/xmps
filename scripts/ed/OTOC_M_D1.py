import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir) 
from fMPS import fMPS
from tdvp.tdvp_fast import MPO_TFI
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body, paulis, N_body_paulis
from numpy import load, linspace, save, sum, log, array, cumsum as cs 
from numpy import arange as ar, mean, eye, diag, max, exp
from numpy.linalg import eigvalsh, svd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensor import H as cT, C as c
from tqdm import tqdm
from tdvp.ex import sympmat, williamson, tr_symp, trace_out
from scipy.sparse.linalg import expm
Sx, Sy, Sz = spins(0.5)

L = 6 
D = 8

S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

ent = Sz1@Sz2
loc = Sx1+Sz1, Sx2+Sz2

listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)
T = linspace(0, 2, 100)
D = 1
#mps = fMPS().load('fixtures/product{}.npy'.format(L))
mps = fMPS().random(L, 2, 1).left_canonicalise()

As = Trajectory(mps,
                H=listH,
                W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).invfreeint(T)

As2 = Trajectory(mps,
                H=listH,
                W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).invfreeint(T)

As_ = Trajectory(mps,
                H=listH,
                W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).edint(T)

δt = T[1]-T[0]
Js = [A.jac(listH) for A in As.mps_list()]
Ms = [eye(Js[0].shape[0])]
print(Ms[0].shape)
for J in Js:
    Ms.append(Ms[-1]@expm(J*δt))
Ms = Ms[1:]

n = Js[0].shape[0]//2
Ω = sympmat(n)
Cs = [(M@Ω)**2 for M in Ms]

def s_max(M):
    """return largest singular value
    """
    s_max = max(svd(M)[1])
    return s_max

def bosonic_renyi(M, tr='half'):
    n = M.shape[0]//2
    if tr is 'half':
        tr = range(n//2, n)
    nu = williamson(trace_out(M.T@M, tr))[0]
    return sum(log(diag(nu)))

def lyapunov(M):
    """lyapunov: returns λ(t)
    """
    s_max = max(svd(M)[1])
    return log(s_max)/2

fig, ax = plt.subplots(n+1, 1, sharex=True, figsize=(7.5, 7.5))
for i, j in zip(range(n), range(n)):
    OTOC  = Trajectory(mps,
                       H=fullH,
                       W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)], fullH=True).ed_OTOC(T, (S_list[i][0], S_list[j][0]))
    ax[i].plot(T, [C[n-i, n-j] for C in Cs], label='bosons')
    ax[i].plot(T, OTOC, label='spins')
    ax[i].legend(loc=1, bbox_to_anchor=(1.2, 1))
    ax[i].set_ylabel('site {}'.format(i+1))
ax[-1].set_xlabel('t')
ax[n].plot(T, As_.renyi(), label='spins')
ax[n].plot(T, As2.renyi(), label='D=2')
ax[n].plot(T, [bosonic_renyi(M) for M in Ms], label='bosons')
ax[n].legend(loc=1, bbox_to_anchor=(1.2, 1))
ax[n].set_ylabel('$R_2$')
fig.suptitle('spin vs boson OTOC, D=1')
plt.tight_layout()
fig.subplots_adjust(top=0.95)
#plt.savefig('images/OTOC/D=1.pdf', bbox_inches='tight')
plt.show()
