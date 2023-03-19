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
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)

GAMMA = 1.0
THETA = 1e-5

num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P 

# 1. initialize an array V(s) = 0 for all s in S+
# and arbitrary pi(s) for all a in A+ for all s in S+
V = np.zeros(num_states)

#Loop
while True:
    #delta <- 0
    delta = 0
    #Loop for each s
    for s in range(num_states):
        # v <- V(s)
        old_value = V[s]
        new_action_values = np.zeros(num_actions)
        #V(s) = max_a(sum(p(s,a)*[r + gamma*v(s')]))
        for a in range(num_actions):
            # sum over s', r
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_]) / num_actions
        V[s] = max(new_action_values)
        #delta <-max(delta|v - V(s)|)
        delta = max(delta, np.abs(old_value - V[s]))
    #until delta < theta
    if delta < THETA:
        break
    
# extract deterministic optimal policy using action value
pi = np.zeros((num_states, num_actions))

for s in range(num_states):
    #pi(s) = argmax_a(sum(p(s,a)*[r + gamma*v(s')]))
    action_values = np.zeros(num_actions)

    for a in range(num_actions):
        # sum over s', r
        for prob, s_, r, _ in transitions[s][a]:
            action_values[a] += prob * (r + GAMMA * V[s_])
            
    #pi(s) <- argmax_a(action_values)
    new_action = np.argmax(action_values)
    pi[s] = np.eye(num_actions)[new_action]

print("Optimal Value = \n", V.reshape(4, 4))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4))
