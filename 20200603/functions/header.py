# header files and constant variables
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import RectBivariateSpline as RS
from multiprocessing import Pool
from functools import partial
from pyswarm import pso
import warnings
warnings.filterwarnings("ignore")
np.printoptions(precision=2)

# time line
T_min = 0
T_max = 70
T_R = 45
# discounting factor
beta = 1/(1+0.02)
# All the money amount are denoted in thousand dollars
earningShock = [0.8,1.2]
# Define transition matrix of economical states
# GOOD -> GOOD 0.8, BAD -> BAD 0.6
Ps = np.array([[0.6, 0.4],[0.2, 0.8]])
# current risk free interest rate
r_f = np.array([0.01 ,0.03])
# stock return depends on current and future econ states
# r_m = np.array([[-0.2, 0.15],[-0.15, 0.2]])
r_m = np.array([[-0.15, 0.20],[-0.15, 0.20]])
# expected return on stock market
# r_bar = 0.0667
r_bar = 0.02
# probability of survival
Pa = np.load("prob.npy")
# deterministic income
detEarning = np.load("detEarning.npy")
# probability of employment transition Pe[s, s_next, e, e_next]
Pe = np.array([[[[0.3, 0.7], [0.1, 0.9]], [[0.25, 0.75], [0.05, 0.95]]],
               [[[0.25, 0.75], [0.05, 0.95]], [[0.2, 0.8], [0.01, 0.99]]]])
# tax rate before and after retirement
tau_L = 0.2
tau_R = 0.1
# minimum consumption
c_bar = 3

#Define the utility function
def u(c):
    gamma = 2
    return (np.float_power(max(c-c_bar,0),1-gamma) - 1)/(1 - gamma)

#Define the bequeath function, which is a function of wealth
def uB(w):
    B = 2
    return B*u(w+c_bar+1)

#Define the earning function, which applies for both employment and unemployment
def y(t, x):
    w, n, s, e, A = x
    if A == 0:
        return 0
    else:
        if t < T_R:
            return detEarning[t] * earningShock[int(s)] * e + (1-e) * 5
        else:
            return detEarning[t]

# Define the reward funtion depends on both state and action.
def R(x, a):
    c, b, k = a
    w, n, s, e, A = x
    if A == 0:
        return uB(w+(1+r_bar)*n)
    else:
        return u(c)
