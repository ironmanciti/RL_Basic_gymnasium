# One-step Actor-Critic(episodic), for estimating pi_theta == pi_*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

env_name = 'CartPole-v1'
# env_name = 'LunarLander-v2'
env = gym.make(env_name)

n_actions = env.action_space.n
action_space = np.arange(env.action_space.n)
    
print(n_actions)

# A differentiable policy parameterization pi(a|s,theta) - policy network
# A differentiable state-value function parameterization v(s,w) - value network
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc_pi = nn.Linear(256, n_actions)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob
    
    def v(self, state):
        x = F.relu(self.fc1(state))
        v = self.fc_v(x)
        return v

# Select step-size parameters 0 < alpha < 1
alpha = 0.001 #0.0002  # 0.001
#Choose discount rate 0<gamma<1
gamma = 0.98
#Chose max number of episodes N
N = 1000  # 10000, 5000

batch_size = 32
# Initialize the parameters theta and state-value weights w
model = ActorCritic(env.observation_space.shape, n_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=alpha)

rendering = True

total_rewards = []

def make_batch(memory):
    batch_states, batch_actions, batch_rewards, batch_next_state, batch_done = [], [], [], [], []
    for transition in memory:
        s, a, r, s_, done = transition
        batch_states.append(s)
        batch_actions.append([a])
        batch_rewards.append([r])
        batch_next_state.append(s_)
        done_mask = 0 if done else 1
        batch_done.append([done_mask])
    return torch.FloatTensor(batch_states).to(device), torch.LongTensor(batch_actions).to(device), \
        torch.FloatTensor(batch_rewards).to(device), torch.FloatTensor(batch_next_state).to(device), \
        torch.FloatTensor(batch_done).to(device)
  
# loop forever(for each episode):
for episode in range(N):
    if episode > N * 0.95:
        env = gym.make(env_name, render_mode='human')
        
    # Initialize S (first state of episode)
    s, _ = env.reset()

    done = False
    memory = []
    # Loop while S is not terminal(for each time step):
    while not done:
        # A ~ pi(.|S,theta) - policy network에서 action 하나를 sampling
        probs = model.pi(torch.tensor(s, dtype=torch.float).to(device)).detach().numpy()
        a = np.random.choice(action_space, p=probs)
        # Take action A, observe S', R
        s_, r, terminated, truncated, _ = env.step(a)
        memory.append((s, a, r, s_, done))
        # S <- S'
        s = s_
        
        done = terminated or truncated
            
        if done:
            s_batch, a_batch, r_batch, s_next_batch, done_batch = make_batch(memory)
            
            # delta <- R + gamma * v(S',w) - v(S,w) (if S' is terminal, then v(S',w) = 0)
            td_target = r_batch + gamma * model.v(s_next_batch) * (1-done_batch)
            # advantage = reward + gamma * v(S',w) - v(S,w) --> advantage = delta
            delta = td_target - model.v(s_batch)
            
            # w <- w + alpha * delta * gradient(v(S,w)) - value network parameter update
            # theta <- theta + alpha * I * delta * gradient(pi(A|S,theta)) - policy network parameter update
            pi = model.pi(s_batch)
            pi_a = pi.gather(1, a_batch)
            # loss = -1 * policy.logprob(action) * advantage + critic loss
            loss = -1 * torch.log(pi_a) * delta + F.smooth_l1_loss(model.v(s_batch), td_target)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            total_rewards.append(sum(r_batch.detach().numpy()))
            
    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        print(f'episode {episode},  최근 100 episode 평균 reward {avg_score: .2f}')
