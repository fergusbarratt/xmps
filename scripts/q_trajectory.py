import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from numpy import array, linspace
import matplotlib.pyplot as plt
from mps_examples import comp_z

tens_0 = array([[1, 0], [0, 0]])
mps_0 = fMPS().left_from_state(tens_0)

T = linspace(0, 10, 100)
exps, lys = Trajectory(mps_0, None).q_trajectory(T, 0.1)
plt.plot(exps)
plt.show()
