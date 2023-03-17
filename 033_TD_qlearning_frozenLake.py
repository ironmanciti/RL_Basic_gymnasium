# Q-Learning (off-policy TD control) for estimating pi=pi*
# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#Algorithm parameters: step size alpha, small e>0
GAMMA = 0.99
ALPHA = 0.9
epsilon = 0.3
num_episodes = 10000 

stochastic = False
if stochastic:
  env = gym.make('FrozenLake-v1', render_mode=None)
else:
  env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
  
num_actions = env.action_space.n 

win_pct = []
scores = []
    
# Initialize Q(s,a) arbitrarily except that Q(terminal, .)=0
Q = defaultdict(lambda: np.zeros(num_actions))

# Loop for each episode:
for episode in range(num_episodes):
    # Initialize S
    s, _ = env.reset()  
    # Loop for each step of episode:
    terminated, truncated = False, False
    score = 0
    while not terminated and not truncated:       
        # Choose A from S using policy derived from Q (eg. e-greedy)
        # behavior policy : e-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else : 
            a = np.argmax(Q[s])
            
        # Take action A, observe R, S'
        s_, r, terminated, truncated, _ = env.step(a)  
        score += r
            
        #Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        #자신이 따르는 정책에 상관없이 최적 행동가치함수 q*를 직접 근사
        # target policy : greedy policy
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])
        # S <-- S'
        s = s_
        # until S is terminal
        
    scores.append(score)
        
    if episode % 1000 and episode > 0.8 * num_episodes:
        average = np.mean(scores[-10:])
        win_pct.append(average)

print("Stochastic" if stochastic else "Deterministic")
print("GAMMA = {}, epsilon = {}, ALPHA = {}".format(GAMMA, epsilon, ALPHA))

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if stochastic else 'Deterministic Env'))
plt.show()

#optimal policy 출력
WIDTH = 4
HEIGHT = 4
GOAL = (3, 3)
actions = ['L', 'D', 'R', 'U']  

opt_policy = []
for i in range(HEIGHT):
    opt_policy.append([])
    for j in range(WIDTH):
        opt_action = Q[i*WIDTH+j].argmax()
        if (i, j) == GOAL:
            opt_policy[i].append("G")
        else:
            opt_policy[i].append(actions[opt_action])

for row in opt_policy:
    print(row)
