# Suntton p.101 
# On-Policy First-Visit MC control(for e-soft policies) for optimum policy pi*

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False)
win = 0
lose = 0
draw = 0
GAMMA = 1  # no discount
#Algorithm parameter: small e > 0
e = 0.2 
num_episodes = 100_000

env = gym.make("Blackjack-v1")
num_actions = env.action_space.n

#Initialize
#pi <- an arbitrary e-soft policy (초기 policy)
#Q(s,a) 초기화 for all s, a
#Returns(s, a) <- empty list for all s, a
pi = defaultdict(lambda: np.ones(num_actions) / num_actions)
Q = defaultdict(lambda: np.zeros(num_actions))
Returns = defaultdict(list)

#Repeat forever (for each episode)
for i in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s, _ = env.reset()
    while True:
        p = pi[s]
        a = np.random.choice(np.arange(len(p)), p=p)  # 0:stick, 1:hit
        s_, r, terminated, truncated, _ = env.step(a)
        #s:(sum_hand(player), dealer open card, usable_ace 보유)
        episode.append((s, a, r))
        if terminated or truncated: 
            # 80% episode 동안 policy 개선된 후 win/lose count
            if i > 0.8 * num_episodes: 
                if r == 1:
                    win += 1
                elif r == -1:
                    lose += 1
                else:
                    draw += 1
            break
        s = s_
        
    #G <- 0
    G = 0
    #Loop for each step of episode, t=T-1, T-2,...0
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        G = GAMMA * G + r
        visited_state_action_pair = []
        #Unless the pair S_t, A_t appears in S_0,A_0 S_1,A_1..S_(t-1),A_(t-1):
            #Append G to Returns(S_t, A_t)
            #Q(S_t,A_t) <- average(Returns(S_t, A_t))
        if (s, a) not in visited_state_action_pair:
            Returns[(s, a)].append(G)
            Q[s][a] = np.mean(Returns[(s, a)])
            visited_state_action_pair.append((s, a))
        
        #A* <- argmax_a Q(S_t,a)
        A_star = np.argmax(Q[s])
        #For all a:
            #pi(a|S_t) <- 1-e + e/|A(S_t)| if a = A*
            #          <- e/|A(St)|        if a != A*
        for a in range(num_actions):
            if a == A_star:
                pi[s][a] = 1 - e + e/num_actions
            else:
                pi[s][a] = e/num_actions
                
    if i % 5000 == 0:
        print(f"episode {i} completed...")

print("win ratio = {:.2f}%".format(win/(0.2 * num_episodes)*100))
print("lose ratio = {:.2f}%".format(lose/(0.2 * num_episodes)*100))
print("draw ratio = {:.2f}%".format(draw/(0.2 * num_episodes)*100))

# prediction
sample_state = (21, 3, True)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, Q[sample_state][optimal_action]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (4, 1, False)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, Q[sample_state][optimal_action]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (14, 8, True)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, Q[sample_state][optimal_action]),
      "stick" if optimal_action == 0 else "hit")
