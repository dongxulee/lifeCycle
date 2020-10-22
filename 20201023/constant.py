import numpy as np
# time line
T_min = 0
T_max = 60
T_R = 45
# discounting factor
beta = 1/(1+0.02)
# utility function parameter 
gamma = 3
# relative importance of housing consumption and non durable consumption 
alpha = 0.8
# parameter used to calculate the housing consumption 
kappa = 0.3
# depreciation parameter 
delta = 0.025
# housing parameter 
chi = 0.3
# uB associated parameter
B = 2
# constant cost 
c_h = 0.5
# social welfare after the unemployment
welfare = 5
# tax rate before and after retirement
tau_L = 0.2
tau_R = 0.1
# number of states S
nS = 27
# number of states variables
nX = 6


# probability of survival
Pa = np.load("constant/prob.npy")
# deterministic income
detEarning = np.load("constant/detEarning.npy")
# Define transition matrix of economical states S
Ps = np.genfromtxt('constant/Ps.csv',delimiter=',')
fix = (np.sum(Ps, axis = 1) - 1)
for i in range(nS):
    for j in range(nS):
        if Ps[i,j] - fix[i] > 0:
            Ps[i,j] = Ps[i,j] - fix[i]
            break
# The possible GDP growth, stock return, bond return
gkfe = np.genfromtxt('constant/gkfe.csv',delimiter=',')

# GDP growth depending on current S state
gGDP = gkfe[:,0]/100
# risk free interest rate depending on current S state 
r_b = gkfe[:,1]/100
# stock return depending on current S state
r_k = gkfe[:,2]/100
# unemployment rate depending on current S state 
Pe = gkfe[:,7:]/100
Pe = Pe[:,::-1]

# some variables associated with 401k amount
r_bar = 0.02
Nt = [np.sum(Pa[t:]) for t in range(T_max-T_min)]
# discounting factor used to calculate the withdraw amount 
Dt = [np.ceil(((1+r_bar)**N - 1)/(r_bar*(1+r_bar)**N)) for N in Nt]
# income fraction goes into 401k 
yi = 0.005

# variable associated with housing and mortgage 
# mortgage rate 
rh = 0.045
# dincounting factor used to calculate the mortgage amount
D = [((1+rh)**N - 1)/(rh*(1+rh)**N) for N in range(T_max-T_min)]
D[0] = 1
# housing unit
H = 1000
# housing price constant 
pt = 2*250/1000
# 30k rent 1000 sf
pr = 2*10/1000

# stock participation maintenance fee
Km = 0.5
# stock participation cost 
Kc = 5