from header import *
# Define the transition of state
def transition(x, a, t):
    '''
        Input: x current state: (w, n, s, e, A)
               a action taken: (c, b, k)
        Output: the next possible states with corresponding probabilities
    '''
    # unpack variable
    c, b, k = a
    w, n, s, e, A = x
    # variables used to collect possible states and probabilities
    x_next = []
    prob_next = []
    # Agent is dead at the end of last period
    if A == 0:
        for s_next in [0,1]:
            x_next.append([0, 0, s_next, 0, 0])
        return np.array(x_next), Ps[int(s)]
    # Agent is alive
    else:
        # variables needed
        N = np.sum(Pa[t:])
        discounting = ((1+r_bar)**N - 1)/(r_bar*(1+r_bar)**N)
        Pat = [1-Pa[t], Pa[t]]
        r_bond = r_f[int(s)]
        # calcualte n_next
        if t < T_R:
            # before retirement agents put 5% of income to 401k
            if e == 1:
                n_next = (n+0.05*y(t,x))*(1+r_bar)
            else:
                n_next = n*(1+r_bar)

            # for potential s_next, e_next and A_next
            for s_next in [0, 1]:
                r_stock = r_m[int(s), s_next]
                w_next =  b*(1+r_bond) + k*(1+r_stock)
                for e_next in [0,1]:
                    for A_next in [0,1]:
                        x_next.append([w_next, n_next, s_next, e_next, A_next])
                        prob_next.append(Ps[int(s),s_next] * Pat[A_next] * Pe[int(s),s_next,int(e),e_next])

        else:
            # after retirement agents withdraw cash from 401k
            n_next = n*(1+r_bar)-n/discounting
            e_next = 0

            # for potential s_next and A_next
            for s_next in [0, 1]:
                r_stock = r_m[int(s), s_next]
                w_next =  b*(1+r_bond) + k*(1+r_stock)
                for A_next in [0,1]:
                    x_next.append([w_next, n_next, s_next, e_next, A_next])
                    prob_next.append(Pat[A_next] * Ps[int(s), s_next])

    return np.array(x_next), np.array(prob_next)
