from scipy.interpolate import interpn
from multiprocessing import Pool
from functools import partial
from constant import *
import warnings
warnings.filterwarnings("ignore")

#Define the utility function
def u(c):
    return (np.float_power(c, 1-gamma) - 1)/(1 - gamma)

#Define the bequeath function, which is a function of wealth
def uB(tb):
    return B*u(tb)

#Calcualte HE 
def calHE(x):
    # the input x is a numpy array 
    # w, n, M, g_lag, e, s, z = x
    HE = (H+(1-chi)*(1-delta)*x[:,3])*pt - x[:,2]
    return HE

#Calculate TB 
def calTB(x):
    # the input x as a numpy array
    # w, n, M, g_lag, e, s, z = x
    TB = x[:,0] + x[:,1] + calHE(x)
    return TB

#The reward function 
def R(x, a):
    '''
    Input:
        state x: w, n, M, g_lag, e, s
        action a: c, b, k, i, q = a which is a np array
    Output: 
        reward value: the length of return should be equal to the length of a
    '''
    w, n, M, g_lag, e, s, z = x
    reward = np.zeros(a.shape[0])
    # actions with improvement 
    i_index = (a[:,4]==1)
    # actions without improvement
    ni_index = (a[:,4]!=1)
    # housing consumption with improvement
    i_h = H + (1-delta)*g_lag + a[i_index][:,3]
    i_Vh = (1+kappa)*i_h
    # housing consumption without improvement
    ni_h = H + (1-delta)*g_lag
    ni_Vh = (1-kappa)*(ni_h-(1-a[ni_index][:,4])*H)
    # combined consumption with and without improvement
    i_C = np.float_power(a[i_index][:,0], alpha) * np.float_power(i_Vh, 1-alpha)
    ni_C = np.float_power(a[ni_index][:,0], alpha) * np.float_power(ni_Vh, 1-alpha)
    reward[i_index] = u(i_C)
    reward[ni_index] = u(ni_C)
    return reward

#Define the earning function, which applies for both employment and unemployment, good econ state and bad econ state 
def y(t, x):
    w, n, M, g_lag, e, s, z = x
    if t <= T_R:
        return detEarning[t] * (1+gGDP[int(s)]) * e + (1-e) * welfare
    else:
        return detEarning[t]

#Earning after tax and fixed by transaction in and out from 401k account 
def yAT(t,x):
    yt = y(t, x)
    w, n, M, g_lag, e, s, z = x
    if t <= T_R and e == 1:
        # yi portion of the income will be put into the 401k 
        return (1-tau_L)*(yt * (1-yi))
    if t <= T_R and e == 0:
        # unemployment
        return yt
    else:
        # t > T_R, n/discounting amount will be withdraw from the 401k 
        return (1-tau_R)*yt + n/Dt[t]

#Define the evolution of the amount in 401k account 
def gn(t, n, x, r_k):
    w, n, M, g_lag, e, s, z = x
    if t <= T_R and e == 1:
        # if the person is employed, then yi portion of his income goes into 401k 
        n_cur = n + y(t, x) * yi
    elif t <= T_R and e == 0:
        # if the perons is unemployed, then n does not change 
        n_cur = n
    else:
        # t > T_R, n/discounting amount will be withdraw from the 401k 
        n_cur = n - n/Dt[t]
        # the 401 grow as the same rate as the stock 
    return (1+r_k)*n_cur 


def transition(x, a, t):
    '''
        Input: state and action and time, where action is an array
        Output: possible future states and corresponding probability 
    '''
    w, n, M, g_lag, e, s, z = x
    s = int(s)
    aSize = len(a)
    # mortgage payment
    m = M/D[T_max-t]
    M_next = M*(1+rh) - m
    # actions
    b = a[:,1]
    k = a[:,2]
    i = a[:,3]
    q = a[:,4]
    z_next = np.ones(aSize)
    g = np.zeros(aSize)
    # transition of z
    if z == 0:
        z_next[k==0] = 0
    # transitin of improvement 
    g[q==1] = (1-delta)*g_lag + i[q==1]
    g[q!=1] = (1-delta)*g_lag
    # we want the output format to be array of all possible future states and corresponding
    # probability. x = [w_next, n_next, M_next, g, e_next, s_next, z_next]

    # create the empty numpy array to collect future states and probability 
    if t >= T_R:
        future_states = np.zeros((aSize*nS,nX))
        n_next = gn(t, n, x, r_k)
        future_states[:,0] = np.repeat(b*(1+r_b[int(s)]), nS) + np.repeat(k, nS)*(1+np.tile(r_k, aSize))
        future_states[:,1] = np.tile(n_next,aSize)
        future_states[:,2] = M_next
        future_states[:,3] = np.repeat(g,nS)
        future_states[:,4] = 0
        future_states[:,5] = np.tile(range(nS),aSize)
        future_states[:,6] = np.repeat(z_next,nS)
        future_probs = np.tile(Ps[s],aSize)
    else:
        future_states = np.zeros((2*aSize*nS,nX))
        n_next = gn(t, n, x, r_k)
        future_states[:,0] = np.repeat(b*(1+r_b[int(s)]), 2*nS) + np.repeat(k, 2*nS)*(1+np.tile(r_k, 2*aSize))
        future_states[:,1] = np.tile(n_next,2*aSize)
        future_states[:,2] = M_next
        future_states[:,3] = np.repeat(g,2*nS)
        future_states[:,4] = np.tile(np.repeat([0,1],nS), aSize)
        future_states[:,5] = np.tile(range(nS),2*aSize)
        future_states[:,6] = np.repeat(z_next,2*nS)
        future_probs = np.tile(np.append(Ps[s]*Pe[s], Ps[s]*(1-Pe[s])),aSize)
    return future_states, future_probs


# Use to approximate the discrete values in V
class Approxy(object):
    def __init__(self, points, Vgrid):
        self.V = Vgrid 
        self.p = points
    def predict(self, xx):
        pvalues = np.zeros(xx.shape[0])
        for e in [0,1]:
            for s in range(nS):
                for z in [0,1]:
                    index = (xx[:,4] == e) & (xx[:,5] == s) & (xx[:,6] == z)
                    pvalues[index]=interpn(self.p, self.V[:,:,:,:,e,s,z], xx[index][:,:4], 
                                           bounds_error = False, fill_value = None)
        return pvalues

# used to calculate dot product
def dotProduct(p_next, uBTB, t):
    if t >= 45:
        return (p_next*uBTB).reshape((len(p_next)//(nS),(nS))).sum(axis = 1)
    else:
        return (p_next*uBTB).reshape((len(p_next)//(2*nS),(2*nS))).sum(axis = 1)
    
# Value function is a function of state and time t < T
def V(x, t, NN):
    w, n, M, g_lag, e, s, z = x
    yat = yAT(t,x)
    m = M/D[T_max - t]
    # If the agent can not pay for the ortgage 
    if yat + w < m:
        return [-1, [0,0,0,0,0]]
    # The agent can pay for the mortgage
    if t == T_max-1:
        # The objective functions of terminal state 
        def obj(actions):
            # Not renting out case 
            # a = [c, b, k, i, q]
            x_next, p_next  = transition(x, actions, t)
            uBTB = uB(calTB(x_next)) # conditional on being dead in the future
            return R(x, actions) + beta * dotProduct(uBTB, p_next, t)
    else:
        def obj(actions):
            # Renting out case
            # a = [c, b, k, i, q]
            x_next, p_next  = transition(x, actions, t)
            V_tilda = NN.predict(x_next) # V_{t+1} conditional on being alive, approximation here
            uBTB = uB(calTB(x_next)) # conditional on being dead in the future
            return R(x, actions) + beta * (Pa[t] * dotProduct(V_tilda, p_next, t) + (1 - Pa[t]) * dotProduct(uBTB, p_next, t))
    
    def obj_solver(obj):
        # Constrain: yat + w - m = c + b + kk + (1+chi)*i*pt + I{i>0}*c_h
        actions = []
        for ip in np.linspace(0.001,0.999,20):
            budget1 = yat + w - m
            if ip*budget1 > c_h:
                i = (budget1*ip - c_h)/((1+chi)*pt)
                budget2 = budget1 * (1-ip)
            else:
                i = 0
                budget2 = budget1
            for cp in np.linspace(0.001,0.999,11):
                c = budget2*cp
                budget3 = budget2 * (1-cp)
                #.....................stock participation cost...............
                for kp in np.linspace(0,1,11):
                    # If z == 1 pay for matainance cost Km = 0.5
                    if z == 1:
                        # kk is stock allocation
                        kk = budget3 * kp
                        if kk > Km:
                            k = kk - Km
                            b = budget3 * (1-kp)
                        else:
                            k = 0
                            b = budget3
                    # If z == 0 and k > 0 payfor participation fee Kc = 5
                    else:
                        kk = budget3 * kp 
                        if kk > Kc:
                            k = kk - Kc
                            b = budget3 * (1-kp)
                        else:
                            k = 0
                            b = budget3
                #..............................................................
                    # q = 1 not renting in this case 
                    actions.append([c,b,k,i,1])
                    
        # Constrain: yat + w - m + (1-q)*H*pr = c + b + kk
        for q in np.linspace(0.001,0.999,20):
            budget1 = yat + w - m + (1-q)*H*pr
            for cp in np.linspace(0.001,0.999,11):
                c = budget1*cp
                budget2 = budget1 * (1-cp)
                #.....................stock participation cost...............
                for kp in np.linspace(0,1,11):
                    # If z == 1 pay for matainance cost Km = 0.5
                    if z == 1:
                        # kk is stock allocation
                        kk = budget2 * kp
                        if kk > Km:
                            k = kk - Km
                            b = budget2 * (1-kp)
                        else:
                            k = 0
                            b = budget2
                    # If z == 0 and k > 0 payfor participation fee Kc = 5
                    else:
                        kk = budget2 * kp 
                        if kk > Kc:
                            k = kk - Kc
                            b = budget2 * (1-kp)
                        else:
                            k = 0
                            b = budget2
                #..............................................................
                    # i = 0, no housing improvement when renting out 
                    actions.append([c,b,k,0,q])            
                               
        actions = np.array(actions)
        values = obj(actions)
        fun = np.max(values)
        ma = actions[np.argmax(values)]
        return fun, ma
    
    fun, action = obj_solver(obj)
    return np.array([fun, action])


# wealth discretization 
ws = np.array([10,25,50,75,100,125,150,175,200,250,500,750,1000,1500,3000])
w_grid_size = len(ws)
# 401k amount discretization 
ns = np.array([1, 5, 10, 15, 25, 40, 65, 100, 150, 300, 400, 1000])
n_grid_size = len(ns)
# Mortgage amount
Ms = np.array([0.01*H,0.05*H,0.1*H,0.2*H,0.3*H,0.4*H,0.5*H,0.6*H,0.7*H,0.8*H]) * pt
M_grid_size = len(Ms)
# Improvement amount 
gs = np.array([0,50,100,200,500,1500])
g_grid_size = len(gs)
points = (ws,ns,Ms,gs)
# dimentions of the state
dim = (w_grid_size, n_grid_size,M_grid_size,g_grid_size,2,nS,2)
dimSize = len(dim)

xgrid = np.array([[w, n, M, g_lag, e, s, z] 
                            for w in ws
                            for n in ns
                            for M in Ms
                            for g_lag in gs 
                            for e in [0,1]
                            for s in range(nS)
                            for z in [0,1]
                            ]).reshape(dim + (dimSize,))

# reshape the state grid into a single line of states to facilitate multiprocessing
xs = xgrid.reshape((np.prod(dim),dimSize))
Vgrid = np.zeros(dim + (T_max,))
cgrid = np.zeros(dim + (T_max,))
bgrid = np.zeros(dim + (T_max,))
kgrid = np.zeros(dim + (T_max,))
igrid = np.zeros(dim + (T_max,))
qgrid = np.zeros(dim + (T_max,))
print("The size of the grid: ", dim + (T_max,))


# value iteration part, create multiprocesses 32
pool = Pool()
for t in range(T_max-1,T_max-5, -1):
    print(t)
    if t == T_max - 1:
        f = partial(V, t = t, NN = None)
        results = np.array(pool.map(f, xs))
    else:
        approx = Approxy(points,Vgrid[:,:,:,:,:,:,:,t+1])
        f = partial(V, t = t, NN = approx)
        results = np.array(pool.map(f, xs))
    Vgrid[:,:,:,:,:,:,:,t] = results[:,0].reshape(dim)
    cgrid[:,:,:,:,:,:,:,t] = np.array([r[0] for r in results[:,1]]).reshape(dim)
    bgrid[:,:,:,:,:,:,:,t] = np.array([r[1] for r in results[:,1]]).reshape(dim)
    kgrid[:,:,:,:,:,:,:,t] = np.array([r[2] for r in results[:,1]]).reshape(dim)
    igrid[:,:,:,:,:,:,:,t] = np.array([r[3] for r in results[:,1]]).reshape(dim)
    qgrid[:,:,:,:,:,:,:,t] = np.array([r[4] for r in results[:,1]]).reshape(dim)
pool.close()

np.save("Vgrid" + str(H), Vgrid)
np.save("cgrid" + str(H), cgrid)
np.save("bgrid" + str(H), bgrid)
np.save("kgrid" + str(H), kgrid)
np.save("igrid" + str(H), igrid)
np.save("qgrid" + str(H), qgrid)