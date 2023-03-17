# Suntton p.101 
# On-Policy First-Visit MC control(for e-soft policies) for optimum policy pi*

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

#state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False)
win = 0
lose = 0
draw = 0
GAMMA = 1  # no discount
#Algorithm parameter: samll e > 0
e = 0.2 
num_episodes = 100_000

env = gym.make("Blackjack-v1")
num_actions = env.action_space.n

#Initialize
#pi <- and arbitrary e-soft policy
#Q(s,a) for all s, a
#Returns(s, a) <- empty list for all s, a
pi = defaultdict(lambda: np.ones(num_actions, dtype=float) / num_actions)
Q = defaultdict(lambda: np.zeros(num_actions))
Returns = defaultdict(list)

#Repeat forever (for each episode)
for _ in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    episode = []
    s, _ = env.reset()
    while True:
        P = pi[s]
        a = np.random.choice(np.arange(len(P)), p=P)  # 0:stick, 1:hit
        s_, r, terminated, truncated, _ = env.step(a)
        #s:(sum_hand(player), dealer open card, usable_ace 보유)
        episode.append((s, a, r))
        if terminated or truncated:
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

print("win ratio = {:.2f}%".format(win/num_episodes*100))
print("lose ratio = {:.2f}%".format(lose/num_episodes*100))
print("draw ratio = {:.2f}%".format(draw/num_episodes*100))
    
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
       
# prediction
sample_state = (21, 3, True)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (4, 1, False)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")

sample_state = (14, 8, True)
optimal_action = np.argmax(Q[sample_state])
print("state {}의 가치 = {:.2f}".format(sample_state, V[sample_state]),
      "stick" if optimal_action == 0 else "hit")
    
X, Y = np.meshgrid(
    np.arange(1, 11),    # dealer가 open 한 card
    np.arange(12, 22))   # player가 가진 card 합계

#V[(sum_hand(player), dealer open card, usable_ace 보유)]
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], False)],
                                    2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], True)],
                                 2, np.dstack([X, Y]))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3),
                               subplot_kw={'projection': '3d'})

ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Dealer open Cards')
ax1.set_ylabel('Player Cards')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Dealer open Cards')
ax0.set_ylabel('Player Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')

plt.show()
