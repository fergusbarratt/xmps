from pymps.iHam import optimise, heis, isin
from numpy import pi

A, e = optimise(isin(0.5), 2, 'unitary')
print(e, -4/pi)
