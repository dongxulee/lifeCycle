from scipy.interpolate import interpn
from constant import * 
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings("ignore")
np.printoptions(precision=2)

#Define the utility function
def u(c):
    # shift utility function to the left, so it only takes positive value
    return (np.float_power(c, 1-gamma) - 1)/(1 - gamma)

#Define the bequeath function, which is a function of wealth
def uB(tb):
    return B*u(tb)

#Calculate TB_rent
def calTB_rent(x):
    # change input x as numpy array
    # w, n, e, s, z = x
    TB = x[:,0] + x[:,1]
    return TB

#Calculate TB_own 
def calTB_own(x):
    # change input x as numpy array
    # transiton from (w, n, e, s, z) -> (w, n, M, e, s, z, H)
    TB = x[:,0] + x[:,1] + x[:,6]*pt - x[:,2]
    return TB

#Reward function for renting
def u_rent(a):
    '''
    Input:
        action a: c, b, k, h = a 
    Output: 
        reward value: the length of return should be equal to the length of a
    '''
    c = a[:,0]
    h = a[:,3]
    C = np.float_power(c, alpha) * np.float_power(h, 1-alpha)
    return u(C)

#Reward function for owning 
def u_own(a):
    '''
    Input:
        action a: c, b, k, M, H = a
    Output: 
        reward value: the length of return should be equal to the length of a
    '''
    c = a[:,0]
    H = a[:,4]
    C = np.float_power(c, alpha) * np.float_power((1+kappa)*H, 1-alpha)
    return u(C)


def transition_to_rent(x,a,t):
    '''
        imput: a is np array constains all possible actions 
        output: from x = [w, n, e, s, z] to x = [w, n, e, s, z]
    '''
    w, n, e, s, z = x
    s = int(s)
    e = int(e)
    nX = len(x)
    aSize = len(a)
    # actions 
    b = a[:,1]
    k = a[:,2]
    h = a[:,3]
    # transition of z
    z_next = np.ones(aSize)
    if z == 0:
        z_next[k==0] = 0
    # transition before T_R and after T_R
    if t >= T_R:
        future_states = np.zeros((aSize*nS,nX))
        n_next = gn(t, n, x, (r_k+r_b)/2)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), nS) + np.repeat(k, nS)*(1+np.tile(r_k, aSize))
        future_states[:,1] = np.tile(n_next,aSize)
        future_states[:,2] = 0
        future_states[:,3] = np.tile(range(nS),aSize)
        future_states[:,4] = np.repeat(z_next,nS)
        future_probs = np.tile(Ps[s],aSize)
    else:
        future_states = np.zeros((2*aSize*nS,nX))
        n_next = gn(t, n, x, (r_k+r_b)/2)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), 2*nS) + np.repeat(k, 2*nS)*(1+np.tile(r_k, 2*aSize))
        future_states[:,1] = np.tile(n_next,2*aSize)
        future_states[:,2] = np.tile(np.repeat([0,1],nS), aSize)
        future_states[:,3] = np.tile(range(nS),2*aSize)
        future_states[:,4] = np.repeat(z_next,2*nS)
        # employed right now:
        if e == 1:
            future_probs = np.tile(np.append(Ps[s]*Pe[s,e], Ps[s]*(1-Pe[s,e])),aSize)
        else:
            future_probs = np.tile(np.append(Ps[s]*(1-Pe[s,e]), Ps[s]*Pe[s,e]),aSize)  
    return future_states, future_probs

def transition_to_own(x,a,t):
    '''
        imput a is np array constains all possible actions 
        from x = [w, n, e, s, z] to x = [w, n, M, e, s, z, H]
    '''
    w, n, e, s, z = x
    s = int(s)
    e = int(e)
    nX = len(x)
    aSize = len(a)
    # actions 
    b = a[:,1]
    k = a[:,2]
    M = a[:,3]
    M_next = M_next*(1+rh)
    H = a[:,4]
    # transition of z
    z_next = np.ones(aSize)
    if z == 0:
        z_next[k==0] = 0
    # transition before T_R and after T_R
    if t >= T_R:
        future_states = np.zeros((aSize*nS,nX))
        n_next = gn(t, n, x, (r_k+r_b)/2)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), nS) + np.repeat(k, nS)*(1+np.tile(r_k, aSize))
        future_states[:,1] = np.tile(n_next,aSize)
        future_states[:,2] = np.repeat(M_next,nS)
        future_states[:,3] = 0
        future_states[:,4] = np.tile(range(nS),aSize)
        future_states[:,5] = np.repeat(z_next,nS)
        future_states[:,6] = np.repeat(H,nS)
        future_probs = np.tile(Ps[s],aSize)
    else:
        future_states = np.zeros((2*aSize*nS,nX))
        n_next = gn(t, n, x, (r_k+r_b)/2)
        future_states[:,0] = np.repeat(b*(1+r_b[s]), 2*nS) + np.repeat(k, 2*nS)*(1+np.tile(r_k, 2*aSize))
        future_states[:,1] = np.tile(n_next,2*aSize)
        future_states[:,2] = np.repeat(M_next,2*nS)
        future_states[:,3] = np.tile(np.repeat([0,1],nS), aSize)
        future_states[:,4] = np.tile(range(nS),2*aSize)
        future_states[:,5] = np.repeat(z_next,2*nS)
        future_states[:,6] = np.repeat(H,2*nS)
        # employed right now:
        if e == 1:
            future_probs = np.tile(np.append(Ps[s]*Pe[s,e], Ps[s]*(1-Pe[s,e])),aSize)
        else:
            future_probs = np.tile(np.append(Ps[s]*(1-Pe[s,e]), Ps[s]*Pe[s,e]),aSize)  
    return future_states, future_probs


class Approxy(object):
    def __init__(self, pointsRent, Vrent, Vown, t):
        self.Vrent = Vrent 
        self.Vown = Vown
        self.Prent = pointsRent
        self.t = t
    def predict(self, xx):
        if xx.shape[1] == 5:
            # x = [w, n, e, s, z]
            pvalues = np.zeros(xx.shape[0])
            for e in [0,1]:
                for s in range(nS):
                    for z in [0,1]: 
                        index = (xx[:,2] == e) & (xx[:,3] == s) & (xx[:,4] == z)
                        pvalues[index]=interpn(self.Prent, self.Vrent[:,:,e,s,z], xx[index][:,:2], 
                                               bounds_error = False, fill_value = None)
            return pvalues
        else: 
            # x = w, n, M, e, s, z, H
            pvalues = np.zeros(xx.shape[0])
            for i in range(len(H_options)):
                H = H_options[i]
                # Mortgage amount, * 0.25 is the housing price per unit
                Ms = np.array([0.01*H,0.05*H,0.1*H,0.2*H,0.3*H,0.4*H,0.5*H,0.8*H]) * pt
                points = (ws,ns,Ms)
                for e in [0,1]:
                    for s in range(nS):
                        for z in [0,1]: 
                            index = (xx[:,3] == e) & (xx[:,4] == s) & (xx[:,5] == z) & (xx[:,6] == H)
                            pvalues[index]=interpn(points, self.Vown[i][:,:,:,e,s,z,self.t], xx[index][:,:3], 
                                                   method = "nearest",bounds_error = False, fill_value = None)
            return pvalues

# used to calculate dot product
def dotProduct(p_next, uBTB, t):
    if t >= T_R:
        return (p_next*uBTB).reshape((len(p_next)//(nS),(nS))).sum(axis = 1)
    else:
        return (p_next*uBTB).reshape((len(p_next)//(2*nS),(2*nS))).sum(axis = 1)
    
# Value function is a function of state and time, according to the restriction transfer from renting to ownning can only happen
# between the age: 0 - 25
def V(x, t, NN):
    w, n, e, s, z = x
    yat = yAT(t,x)
    
    # first define the objective function solver and then the objective function
    def obj_solver_rent(obj_rent):
        # a = [c, b, k, h] 
        # Constrain: yat + w = c + b + k + pr*h
        actions = []
        for hp in np.linspace(0.001,0.999,20):
            budget1 = yat + w
            h = budget1 * hp/pr
            budget2 = budget1 * (1-hp)
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
                    actions.append([c,b,k,h])
                    
        actions = np.array(actions)
        values = obj_rent(actions)
        fun = np.max(values)
        ma = actions[np.argmax(values)]
        return fun, ma          
                    
    def obj_solver_own(obj_own):
    # a = [c, b, k, M, H]
    # possible value of H = {750, 1000, 1500, 2000} possible value of [0.2H, 0.5H, 0.8H]]*pt
    # (M, t, rh) --> m 
    # Constrain: yat + w = c + b + k + (H*pt - M) + ch
        actions = []
        for H in H_options:
            for mp in M_options:
                M = mp*H*pt
                m = M/D[T_max - t]
                # 5 is the welfare income which is also the minimum income
                if (H*pt - M) + c_h <= yat + w and m < pr*H + 5:
                    budget1 = yat + w - (H*pt - M) - c_h
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
                            actions.append([c,b,k,M,H])
                            
        if len(actions) == 0:
            return -np.inf, [0,0,0,0,0]
        else:
            actions = np.array(actions)
            values = obj_own(actions)
            fun = np.max(values)
            ma = actions[np.argmax(values)]
            return fun, ma
    
    if t == T_max-1:
        # The objective function of renting
        def obj_rent(actions): 
            # a = [c, b, k, h]
            x_next, p_next  = transition_to_rent(x, actions, t)
            uBTB = uB(calTB_rent(x_next)) 
            return u_rent(actions) + beta * dotProduct(uBTB, p_next, t) 

        fun, action = obj_solver_rent(obj_rent)
        return np.array([fun, action])
    
    # If the agent is older that 25 or if the agent is unemployed then keep renting 
    elif t > 25 or e == 0:
        # The objective function of renting
        def obj_rent(actions):
            # a = [c, b, k, h]
            x_next, p_next  = transition_to_rent(x, actions, t)
            V_tilda = NN.predict(x_next) # V_rent_{t+1} used to approximate, shape of x is [w,n,e,s]
            uBTB = uB(calTB_rent(x_next))
            return u_rent(actions) + beta * (Pa[t] * dotProduct(V_tilda, p_next, t) + (1 - Pa[t]) * dotProduct(uBTB, p_next, t))

        fun, action = obj_solver_rent(obj_rent)
        return np.array([fun, action])
    # If the agent is younger that 45 and agent is employed. 
    else:
        # The objective function of renting
        def obj_rent(actions):
            # a = [c, b, k, h]
            x_next, p_next  = transition_to_rent(x, actions, t)
            V_tilda = NN.predict(x_next) # V_rent_{t+1} used to approximate, shape of x is [w,n,e,s]
            uBTB = uB(calTB_rent(x_next))
            return u_rent(actions) + beta * (Pa[t] * dotProduct(V_tilda, p_next, t) + (1 - Pa[t]) * dotProduct(uBTB, p_next, t))
        # The objective function of owning
        def obj_own(actions):
            # a = [c, b, k, M, H]
            x_next, p_next  = transition_to_own(x, actions, t)
            V_tilda = NN.predict(x_next) # V_own_{t+1} used to approximate, shape of x is [w, n, M, e, s, H]
            uBTB = uB(calTB_own(x_next))
            return u_own(actions) + beta * (Pa[t] * dotProduct(V_tilda, p_next, t) + (1 - Pa[t]) * dotProduct(uBTB, p_next, t))

        fun1, action1 = obj_solver_rent(obj_rent)
        fun2, action2 = obj_solver_own(obj_own)
        if fun1 > fun2:
            return np.array([fun1, action1])
        else:
            return np.array([fun2, action2])
        
        
        
# wealth discretization 
ws = np.array([10,25,50,75,100,125,150,175,200,250,500,750,1000,1500,3000])
w_grid_size = len(ws)
# 401k amount discretization 
ns = np.array([1, 5, 10, 15, 25, 50, 100, 150, 400, 1000])
n_grid_size = len(ns)
pointsRent = (ws, ns)
# dimentions of the state
dim = (w_grid_size, n_grid_size, 2, nS, 2)
dimSize = len(dim)

xgrid = np.array([[w, n, e, s, z]
                            for w in ws
                            for n in ns
                            for e in [0,1]
                            for s in range(nS)
                            for z in [0,1]
                            ]).reshape(dim + (dimSize,))

xs = xgrid.reshape((np.prod(dim),dimSize))

Vgrid = np.zeros(dim + (T_max,))
cgrid = np.zeros(dim + (T_max,))
bgrid = np.zeros(dim + (T_max,))
kgrid = np.zeros(dim + (T_max,))
hgrid = np.zeros(dim + (T_max,))
# Policy function of buying a house 
Mgrid = np.zeros(dim + (T_max,))
Hgrid = np.zeros(dim + (T_max,))

# # Define housing choice part: Housing unit options and Mortgage amount options
V1000 = np.load("Vgrid1000.npy")
V1500 = np.load("Vgrid1500.npy")
V2000 = np.load("Vgrid2000.npy")
V750 = np.load("Vgrid750.npy")
H_options = [750, 1000, 1500, 2000]
M_options = [0.2, 0.5, 0.8]
Vown = [V750, V1000, V1500, V2000]
print("The size of the grid: ", dim + (T_max,))

# value iteration part 
pool = Pool()
for t in range(T_max-1,T_min, -1):
    print(t)
    if t == T_max - 1:
        f = partial(V, t = t, NN = None)
        results = np.array(pool.map(f, xs))
    else:
        approx = Approxy(pointsRent,Vgrid[:,:,:,:,:,t+1], Vown, t+1)
        f = partial(V, t = t, NN = approx)
        results = np.array(pool.map(f, xs))
    Vgrid[:,:,:,:,:,t] = results[:,0].reshape(dim)
    cgrid[:,:,:,:,:,t] = np.array([r[0] for r in results[:,1]]).reshape(dim)
    bgrid[:,:,:,:,:,t] = np.array([r[1] for r in results[:,1]]).reshape(dim)
    kgrid[:,:,:,:,:,t] = np.array([r[2] for r in results[:,1]]).reshape(dim)
    # if a = [c, b, k, h]
    hgrid[:,:,:,:,:,t] = np.array([r[3] if len(r) == 4 else r[4] for r in results[:,1]]).reshape(dim)
    # if a = [c, b, k, M, H]
    Mgrid[:,:,:,:,:,t] = np.array([r[3] if len(r) == 5 else 0 for r in results[:,1]]).reshape(dim)
    Hgrid[:,:,:,:,:,t] = np.array([r[4] if len(r) == 5 else 0 for r in results[:,1]]).reshape(dim)
pool.close()

np.save("Vgrid_renting",Vgrid) 
np.save("cgrid_renting",cgrid) 
np.save("bgrid_renting",bgrid) 
np.save("kgrid_renting",kgrid) 
np.save("hgrid_renting",hgrid) 
np.save("Mgrid_renting",Mgrid) 
np.save("Hgrid_renting",Hgrid) 