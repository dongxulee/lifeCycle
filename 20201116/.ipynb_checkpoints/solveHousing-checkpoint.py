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
    # w, n, M, e, s, z = x
    HE = H*pt - x[:,2]
    return HE

#Calculate TB 
def calTB(x):
    # the input x as a numpy array
    # w, n, M, e, s, z = x
    TB = x[:,0] + x[:,1] + calHE(x)
    return TB

#The reward function 
def R(x, a):
    '''
    Input:
        state x: w, n, M, e, s, z
        action a: c, b, k, q = a which is a np array
    Output: 
        reward value: the length of return should be equal to the length of a
    '''
    w, n, M, e, s, z = x
    reward = np.zeros(a.shape[0])
    # actions with not renting out 
    nrent_index = (a[:,3]==1)
    # actions with renting out 
    rent_index = (a[:,3]!=1)
    # housing consumption not renting out 
    nrent_Vh = (1+kappa)*H
    # housing consumption renting out 
    rent_Vh = (1-kappa)*H*a[:,3]
    # combined consumption with housing consumption 
    nrent_C = np.float_power(a[nrent_index][:,0], alpha) * np.float_power(nrent_Vh, 1-alpha)
    rent_C = np.float_power(a[rent_index][:,0], alpha) * np.float_power(rent_Vh, 1-alpha)
    reward[nrent_index] = u(nrent_C)
    reward[rent_index] = u(rent_C)
    return reward


def transition(x, a, t):
    '''
        Input: state and action and time, where action is an array
        Output: possible future states and corresponding probability 
    '''
    w, n, M, e, s, z = x
    s = int(s)
    e = int(e)
    aSize = len(a)
    nX = len(x)
    # mortgage payment
    m = M/D[T_max-t]
    M_next = M*(1+rh) - m
    # actions
    b = a[:,1]
    k = a[:,2]
    q = a[:,3]
    # transition of z
    z_next = np.ones(aSize)
    if z == 0:
        z_next[k==0] = 0
    # we want the output format to be array of all possible future states and corresponding
    # probability. x = [w_next, n_next, M_next, e_next, s_next, z_next]
    # create the empty numpy array to collect future states and probability 
    if t >= T_R:
        future_states = np.zeros((aSize*nS,nX))
        n_next = gn(t, x, r_k)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), nS) + np.repeat(k, nS)*(1+np.tile(r_k, aSize))
        future_states[:,1] = np.tile(n_next,aSize)
        future_states[:,2] = M_next
        future_states[:,3] = 0
        future_states[:,4] = np.tile(range(nS),aSize)
        future_states[:,5] = np.repeat(z_next,nS)
        future_probs = np.tile(Ps[s],aSize)
    else:
        future_states = np.zeros((2*aSize*nS,nX))
        n_next = gn(t, x, r_k)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), 2*nS) + np.repeat(k, 2*nS)*(1+np.tile(r_k, 2*aSize))
        future_states[:,1] = np.tile(n_next,2*aSize)
        future_states[:,2] = M_next
        future_states[:,3] = np.tile(np.repeat([0,1],nS), aSize)
        future_states[:,4] = np.tile(range(nS),2*aSize)
        future_states[:,5] = np.repeat(z_next,2*nS)
        # employed right now:
        if e == 1:
            future_probs = np.tile(np.append(Ps[s]*Pe[s,e], Ps[s]*(1-Pe[s,e])),aSize)
        else:
            future_probs = np.tile(np.append(Ps[s]*(1-Pe[s,e]), Ps[s]*Pe[s,e]),aSize)
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
                    index = (xx[:,3] == e) & (xx[:,4] == s) & (xx[:,5] == z)
                    pvalues[index]=interpn(self.p, self.V[:,:,:,e,s,z], xx[index][:,:3], 
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
    w, n, M, e, s, z = x
    yat = yAT(t,x)
    m = M/D[T_max - t]
    # If the agent can not pay for the ortgage 
    if yat + w < m:
        return [0, [0,0,0,0,0]]
    # The agent can pay for the mortgage
    if t == T_max-1:
        # The objective functions of terminal state 
        def obj(actions):
            # Not renting out case 
            # a = [c, b, k, q]
            x_next, p_next  = transition(x, actions, t)
            uBTB = uB(calTB(x_next)) # conditional on being dead in the future
            return R(x, actions) + beta * dotProduct(uBTB, p_next, t)
    else:
        def obj(actions):
            # Renting out case
            # a = [c, b, k, q]
            x_next, p_next  = transition(x, actions, t)
            V_tilda = NN.predict(x_next) # V_{t+1} conditional on being alive, approximation here
            uBTB = uB(calTB(x_next)) # conditional on being dead in the future
            return R(x, actions) + beta * (Pa[t] * dotProduct(V_tilda, p_next, t) + (1 - Pa[t]) * dotProduct(uBTB, p_next, t))
    
    def obj_solver(obj):
        # Constrain: yat + w - m = c + b + kk
        actions = []
        budget1 = yat + w - m
        for cp in np.linspace(0.001,0.999,11):
            c = budget1 * cp
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
                # q = 1 not renting in this case 
                actions.append([c,b,k,1])
                    
        # Constrain: yat + w - m + (1-q)*H*pr = c + b + kk
        for q in np.linspace(0.001,0.999,6):
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
                    actions.append([c,b,k,q])            
                               
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
ns = np.array([1, 5, 10, 15, 25, 50, 100, 150, 400, 1000])
n_grid_size = len(ns)
# Mortgage amount
Ms = np.array([0.01*H,0.05*H,0.1*H,0.2*H,0.3*H,0.4*H,0.5*H,0.8*H]) * pt
M_grid_size = len(Ms)
points = (ws,ns,Ms)
# dimentions of the state
dim = (w_grid_size, n_grid_size,M_grid_size,2,nS,2)
dimSize = len(dim)

xgrid = np.array([[w, n, M, e, s, z] 
                            for w in ws
                            for n in ns
                            for M in Ms
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
qgrid = np.zeros(dim + (T_max,))
print("The size of housing unit: ", H)
print("The size of the grid: ", dim + (T_max,))


# value iteration part, create multiprocesses 32
pool = Pool()
for t in range(T_max-1,T_min, -1):
    print(t)
    if t == T_max - 1:
        f = partial(V, t = t, NN = None)
        results = np.array(pool.map(f, xs))
    else:
        approx = Approxy(points,Vgrid[:,:,:,:,:,:,t+1])
        f = partial(V, t = t, NN = approx)
        results = np.array(pool.map(f, xs))
    Vgrid[:,:,:,:,:,:,t] = results[:,0].reshape(dim)
    cgrid[:,:,:,:,:,:,t] = np.array([r[0] for r in results[:,1]]).reshape(dim)
    bgrid[:,:,:,:,:,:,t] = np.array([r[1] for r in results[:,1]]).reshape(dim)
    kgrid[:,:,:,:,:,:,t] = np.array([r[2] for r in results[:,1]]).reshape(dim)
    qgrid[:,:,:,:,:,:,t] = np.array([r[3] for r in results[:,1]]).reshape(dim)
pool.close()

np.save("Vgrid" + str(H), Vgrid)
np.save("cgrid" + str(H), cgrid)
np.save("bgrid" + str(H), bgrid)
np.save("kgrid" + str(H), kgrid)
np.save("qgrid" + str(H), qgrid)
