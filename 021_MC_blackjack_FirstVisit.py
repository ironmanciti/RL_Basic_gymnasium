# Suntton p.92 First-Visit MC predictions, for estimating V ~ v_pi
# card 조합에 따른 State Value estimate

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

stick_threshold = 19
win_cnt = 0
lose_cnt = 0
draw_cnt = 0
num_episodes = 100_000
GAMMA = 1  # no discount

env = gym.make("Blackjack-v1")

# Input: a policy pi to be evaluated
def pi(state):
    # state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False)
    # player card 가 stick_threshold 이상이면 무조건 stick 하고
    # else 이면 hit 하는 전략
    #0:stick, 1:hit
    return 0 if state[0] >= stick_threshold else 1  

# Initialize V(s)
# Returns(s) <- empty list for all s
V = defaultdict(float)
Returns = defaultdict(list)

# Loop forever(for each episode)
for i in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s, _ = env.reset()
    while True:
        a = pi(s)  # 정책 pi 를 따름
        s_, r, terminated, truncated, _ = env.step(a)
        # s_ : (sum_hand(player), dealer open card, usable_ace 보유)
        episode.append((s, a, r))
        if terminated or truncated:
            if r == 1:
                win_cnt += 1
            elif r == -1:
                lose_cnt += 1
            else:
                draw_cnt += 1
            break
        s = s_
    #G <- 0
    G = 0
    #Loop for each step of spisode: t=T-1,T-2,...,0
    for s, a, r in episode[::-1]:
        #G <- gamma*G + R_(t+1)
        G = GAMMA * G + r  
        visited_states = []
        #Unless S_t appears in S_0, S_1,...S_(t-1):          
        if s not in visited_states:
            #Append G to Returns(S_t)
            Returns[s].append(G)
            #V(S_t) <- average(Returns(S_t))
            V[s] = np.mean(Returns[s])
            visited_states.append(s)    
            
    if i % 5000 == 0:
        print(f"episode {i} completed...") 

print('stick threshold = {}'.format(stick_threshold))
print("win ratio = {:.2f}%".format(win_cnt/num_episodes*100))
print("lose ratio = {:.2f}%".format(lose_cnt/num_episodes*100))
print("draw ratio = {:.2f}%".format(draw_cnt/num_episodes*100))

sample_state = (21, 3, True)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
print(f"     player가 손에 {sample_state[0]}를 들고 dealer가 {sample_state[1]}를 보여주고 있을 때")
sample_state = (14, 1, False)
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]))
print(f"     player가 손에 {sample_state[0]}를 들고 dealer가 {sample_state[1]}를 보여주고 있을 때")
  