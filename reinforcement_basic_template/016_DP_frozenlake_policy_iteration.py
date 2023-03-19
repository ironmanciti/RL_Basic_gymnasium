#Policy Iteration
"""
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

nA = 4
nS = 4*4 = 16
P = {s: {a: [] for a in range(nA)} for s in range(nS)}
env.P[0][0] 
{0: {0: [(0.3333333333333333, 0, 0.0, False), --> (P[s'], s', r, done)
         (0.3333333333333333, 0, 0.0, False),
         (0.3333333333333333, 4, 0.0, False)],
"""


# 1. initialize an array V(s) = 0 for all s in S+
# and arbitrary pi(s) for all a in A+ for all s in S+



while not policy_stable:
    #2. Policy Evaluation
    while True:
        #delta <- 0
        
        #Loop for each s
        for s in range(num_states):
            
            #V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            
            #delta <-max(delta|v - V(s)|)
            
        #until delta < theta
        

    #3. Policy Improvement
    #policy_stable <- true
    
    old_pi = pi
    #For each s:
    for s in range(num_states):
        # pi_s <- argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
        


    #If policy-stable. then stop and return V and pi; else go to 2.

