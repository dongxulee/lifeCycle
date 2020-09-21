import numpy as np
# time line
T_min = 0
T_max = 70
T_R = 45
# discounting factor
beta = 1/(1+0.02)
# utility function parameter 
gamma = 2
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
# All the money amount are denoted in thousand dollars
earningShock = [0.8,1.2]
# Define transition matrix of economical states
# GOOD -> GOOD 0.8, BAD -> BAD 0.6
Ps = np.array([[0.6, 0.4],[0.2, 0.8]])
# current risk free interest rate
r_b = np.array([0.03 ,0.05])
# stock return depends on current and future econ states
r_k = np.array([[-0.2, 0.15],[-0.15, 0.20]])
# expected return on stock market, use to calculate 401k dynamics
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
# constant state variables: Purchase value 250k, down payment 50k, mortgage 200k, interest rate 3.6%,
# 55 payment period, 8.4k per period. One housing unit is roughly 1 square feet. Housing price 0.25k/sf 
# some variables associate with 401k amount
Nt = [np.sum(Pa[t:]) for t in range(T_max-T_min)]
Dt = [np.ceil(((1+r_bar)**N - 1)/(r_bar*(1+r_bar)**N)) for N in Nt]
# income fraction goes into 401k 
yi = 0.005
# mortgate rate 
rh = 0.036
D = [((1+rh)**N - 1)/(rh*(1+rh)**N) for N in range(T_max-T_min)]
D[0] = 1
# housing unit
H = 1000
# housing price constant 
pt = 2*250/1000
# 30k rent 1000 sf
pr = 2*30/1000

# stock participation maintenance fee
Km = 0.5
# stock participation cost 
Kc = 5