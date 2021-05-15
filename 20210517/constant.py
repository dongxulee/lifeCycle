import numpy as np
import jax.numpy as jnp
# time line
T_min = 0
T_max = 60
T_R = 45
# discounting factor
beta = 1/(1+0.02)
# utility function parameter 
gamma = 4
# relative importance of housing consumption and non durable consumption 
alpha = 0.7
# parameter used to calculate the housing consumption 
kappa = 0.3
# uB associated parameter
B = 2
# constant cost 
c_h = 5
c_s = 75
# social welfare after the unemployment
welfare = 20
# tax rate before and after retirement
tau_L = 0.2
tau_R = 0.1
# number of states S
nS = 8
# number of states e
nE = 2
# housing state
nO = 2


'''
    Economic state calibration 
'''

# probability of survival
Pa = jnp.array(np.load("constant/prob.npy"))
# deterministic income
detEarning = jnp.array(np.load("constant/detEarningHigh.npy"))
# rescale the deterministic income
detEarning = detEarning
# Define transition matrix of economical states S
Ps = np.genfromtxt('constant/Ps.csv',delimiter=',')
fix = (np.sum(Ps, axis = 1) - 1)
for i in range(nS):
    for j in range(nS):
        if Ps[i,j] - fix[i] > 0:
            Ps[i,j] = Ps[i,j] - fix[i]
            break
Ps = jnp.array(Ps)
# The possible GDP growth, stock return, bond return
gkfe = np.genfromtxt('constant/gkfe.csv',delimiter=',')
gkfe = jnp.array(gkfe)
# GDP growth depending on current S state
gGDP = gkfe[:,0]/100
# risk free interest rate depending on current S state 
r_b = gkfe[:,1]/100
# stock return depending on current S state
r_k = gkfe[:,2]/100
# unemployment rate depending on current S state 
Pe = gkfe[:,7:]/100
Pe = Pe[:,::-1]

'''
    401k related constants
'''
# some variables associated with 401k amount
r_bar = 0.02
Pa = Pa[:T_max]
Nt = [np.sum(Pa[t:]) for t in range(T_min,T_max)]
#Factor used to calculate the withdraw amount 
Dn = [(r_bar*(1+r_bar)**N)/((1+r_bar)**N - 1) for N in Nt]
Dn[-1] = 1
Dn = jnp.array(Dn)
# income fraction goes into 401k 
yi = 0.05



'''
    housing related constants
'''
# variable associated with housing and mortgage 
# mortgage rate 
rh = 0.045
# housing unit
H = 700
# housing price constant 
pt = 2*250/1000
# 30k rent 1000 sf
pr = 2*10/1000 * 2 
# Dm is used to update the mortgage payment
Dm = [(1+rh) - rh*(1+rh)**(T_max - t)/((1+rh)**(T_max-t)-1) for t in range(T_min, T_max)]
Dm[-1] = 0
Dm = jnp.array(Dm)

# 30 year mortgage
Ms = []
M = H*pt*0.8
m = M*(1+rh) - Dm[30]*M
for i in range(30, T_max):
    M = M*(1+rh) - m
    Ms.append(M)
Ms[-1] = 0
Ms = jnp.array(Ms)


# stock transaction fee
Kc = 0.001


'''
    Discretize the state space
    Discretize the action space 
'''
# actions dicretization(hp, cp, kp)
numGrid = 20
As = np.array(np.meshgrid(np.linspace(0.001,0.999,numGrid), np.linspace(0,1,numGrid), [0,1])).T.reshape(-1,3)
As = jnp.array(As)
# wealth discretization 
ws = np.linspace(0, 400, 20)
ns = np.linspace(0, 300, 10)
ms = np.linspace(0, 0.8*H*pt*(1+rh), 10)
# scales associated with discretization
scaleW = ws.max()/ws.size
scaleN = ns.max()/ns.size
scaleM = ms.max()/ms.size

# dimentions of the state
dim = (ws.size, ns.size, ms.size, nS, nE, nO)
dimSize = len(dim)

xgrid = np.array([[w,n,m,s,e,o] for w in ws
                            for n in ns
                            for m in ms
                            for s in range(nS)
                            for e in range(nE)
                            for o in range(nO)]).reshape(dim + (dimSize,))

Xs = xgrid.reshape((np.prod(dim),dimSize))
Xs = jnp.array(Xs)

Vgrid = np.zeros(dim + (T_max,))
cgrid = np.zeros(dim + (T_max,))
bgrid = np.zeros(dim + (T_max,))
kgrid = np.zeros(dim + (T_max,))
hgrid = np.zeros(dim + (T_max,))
agrid = np.zeros(dim + (T_max,))