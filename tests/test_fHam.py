from pymps.fHam import optimise, isin, heis

L, D = 5, 2
for L in range(2, 10):
    print(optimise(isin(L), L, D)[1])
