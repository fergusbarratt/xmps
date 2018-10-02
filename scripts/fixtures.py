import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from fMPS import fMPS
from spin import spins
Sx, Sy, Sz = spins(0.5)

A = fMPS().load('fixtures/product11.npy')
print(A.Es((Sx, Sy, Sz), 0))
