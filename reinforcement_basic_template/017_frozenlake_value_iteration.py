#Value Iteration
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


#Loop
while True:
    #delta <- 0
    
    #Loop for each s
    for s in range(num_states):
        # v <- V(s)
        
        #V(s) = max_a(sum(p(s,a)*[r + gamma*v(s')]))
        
        #delta <-max(delta|v - V(s)|)
        
    #until delta < theta
    

# extract deterministic optimal policy using action value

for s in range(num_states):
    #pi(s) = argmax_a(sum(p(s,a)*[r + gamma*v(s')]))
    

    #pi(s) <- argmax_a(action_values)
    