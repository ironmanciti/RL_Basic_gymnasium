#Iterative Policy Evaluation
#One-Array version
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
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)

GAMMA = 1.0
THETA = 1e-5
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P 

#Input pi, the policy to be evaluated 
policy = np.ones([num_states, num_actions]) * 0.25

# initialize an array V(s) = 0 for all s in S+
V = np.zeros(num_states)

#Loop
while True:
    #delta <- 0
    delta = 0
    #Loop for each s:
    for s in range(num_states):
        #v <- V(s)
        old_value = V[s]
        new_value = 0
        #update rule : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
        for a, prob_action in enumerate(policy[s]):
            # sum over s', r
            for prob, s_, reward, _ in transitions[s][a]:
                new_value += prob_action * prob * (reward + GAMMA * V[s_])
        V[s] = new_value
        #delta <- max(delta|v - V(s)|)
        delta = max(delta, np.abs(old_value - V[s]))
        
    #until delta < theta
    if delta < THETA:
        break
#V는 v_pi에 수렴
print("수렴한 Optimal Value = \n", V.reshape(4, 4))
