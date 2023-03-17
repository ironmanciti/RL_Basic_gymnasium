#Tabular TD(0) for estimating V_pi
import numpy as np
import gymnasium as gym
from collections import defaultdict

env = gym.make('CartPole-v1', render_mode=None)
gamma = 0.99
num_episodes = 10000
#Tabular 형식으로 해결하기 위해 continuous state를 이산화
states = np.linspace(-0.2094, 0.2094, 10)

#Input: the policy pi to be evaluated
def pi(state):
    #state < 5 면 왼쪽으로 이동, 아니면 오른쪽 이동
    action = 0 if state < 5 else 1
    return action
#Algorithm parameter: step size alpha (0, 1]
alpha = 0.1
#Initialize V(s), arbitrarily except V(terminal)=0
V = defaultdict(float)

#Loop for each episode:
for episode in range(num_episodes):
    #Initialize s
    observation, _ = env.reset()
    s = np.digitize(observation[2], states)
    terminated, truncated = False, False
    #Loop for each step of episode
    while not terminated and not truncated:
        #A <- action given by pi for s
        a = pi(s)
        #Take action A, observe R, S'
        observation_, r, terminated, truncated, _ = env.step(a)
        s_ = np.digitize(observation_[2], states)
        #V(s) <- V(s) + alpha[r+gamma*V(s')-V(s)]
        V[s] += alpha*(r + gamma * V[s_] - V[s])
        #S <- S'
        s = s_
        
    if episode % 1000 == 0:  
        print(f"Episode {episode}")
        for s, v in sorted(V.items()):
            print(f"\tState {s} Value = {v:.2f}")

print("states = ", states)

for s, v in sorted(V.items()):
    print("State Value of {} = {:.2f}".format(s, v))
        
        
