# REINFORCE Algorithm
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical;
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)

action_space = np.arange(env.action_space.n)

# A differentiable policy parameterization pi(a|s,theta)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

#Initialize the parameters theta
pi = PolicyNetwork(input_dims=env.observation_space.shape, n_actions=env.action_space.n).to(device)

#Select step-size parameters 0<alpha<1
alpha = 0.001      #0.001 
#Choose discount rate 0<gamma<1
gamma = 0.99
#Chose max number of episodes N
N = 5000        #10000
# Choose number of episodes to batch together for an update K >= 1
batch_size = 32

optimizer = optim.Adam(pi.parameters(), lr=alpha)

rendering = True

total_rewards = []

def discount_rewards(rewards):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    # return r - r.mean()
    return r 

start_time = time.time()
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1

# While episode n < N do: (training의 안정성 제고)
for episode in range(N):
    if episode > N * 0.95:
        env = gym.make(ENV_NAME, render_mode='human')
    s, _ = env.reset()

    # for K batches do:
    states = []
    rewards = []
    actions = []

    terminated, truncated = False, False
    # Generate an episode s0, a0, r0,...st,at,rt following policy pi(a|s,theta)
    while not terminated and not truncated:
        probs = pi(torch.from_numpy(s).float().to(device)).detach().cpu().numpy()
        a = np.random.choice(action_space, p=probs)
       
        s_, r, terminated, truncated, _ = env.step(a)
        
        states.append(s)
        rewards.append(r)
        actions.append(a)
        done = terminated or truncated
  
        s = s_
        
        if done:
            # for each step in the eposide(t), discount reward do:
            # G_t = sum from t=1 to t=T {gamma^t * R_t}
            batch_rewards.extend(discount_rewards(rewards))
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_counter += 1
            total_rewards.append(sum(rewards))
        
            # If batch is complete, update network
            if batch_counter == batch_size:    
                state_tensor = torch.FloatTensor(batch_states).to(device)
                reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                action_tensor = torch.LongTensor(batch_actions).to(device)
                
                # Calculate policy loss for all episides in the batch L(theta) = -1/m sum(ln(G_t pi(a|s,theta)))))
                log_prob = torch.log(pi(state_tensor))
                selected_log_probs = reward_tensor * \
                        torch.gather(log_prob, 1, action_tensor.unsqueeze(1)).squeeze()
                loss = -1 * selected_log_probs.mean()
    
                # Update the policy: theta <- theta + alpha * grad[L(theat)]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_rewards = []
                batch_actions = []
                batch_states = []
                batch_counter = 1
                
    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        print(f'episode {episode},  최근 100 episode 평균 reward {avg_score: .2f}')

env.close()
print("duration = ", (time.time() - start_time) / 60, "minutes")

running_avg = np.zeros(len(total_rewards))

for i in range(len(running_avg)):
    running_avg[i] = np.mean(total_rewards[max(0, i-100):(i+1)])
plt.plot(running_avg)
plt.title('Running average of previous 100 rewards')
plt.show()
