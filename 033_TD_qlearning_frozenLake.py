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
n_episodes = 10_000 

is_slippery = False

env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
  
num_actions = env.action_space.n 

win_pct = []
scores = []
    
# Initialize Q(s,a) arbitrarily except that Q(terminal, .)=0
Q = defaultdict(lambda: np.zeros(num_actions))

# Loop for each episode:
for episode in range(n_episodes):
    # if episode > n_episodes * 0.995:
    #     env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode="human")
    # Initialize S
    s, _ = env.reset()  
    # Loop for each step of episode:
    score = 0
    while True:       
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
        
        if terminated or truncated:
            break
        
        # S <-- S'
        s = s_
        # until S is terminal
        
    scores.append(score)
        
    if episode % 1000 and episode > 0.8 * n_episodes:
        average = np.mean(scores[-10:])
        win_pct.append(average)

print("Stochastic" if is_slippery else "Deterministic")
print("GAMMA = {}, epsilon = {}, ALPHA = {}".format(GAMMA, epsilon, ALPHA))

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if is_slippery else 'Deterministic Env'))
plt.show()

#optimal policy 출력
WIDTH = 4
HEIGHT = 4
GOAL = (3, 3)
actions = ['L', 'D', 'R', 'U']  

optimal_policy = []
for i in range(HEIGHT):
    optimal_policy.append([])
    for j in range(WIDTH):
        optimal_action = Q[i*WIDTH+j].argmax()
        if (i, j) == GOAL:
            optimal_policy[i].append("G")
        else:
            optimal_policy[i].append(actions[optimal_action])

for row in optimal_policy:
    print(row)
