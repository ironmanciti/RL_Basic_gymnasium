# Deep Q-learning Target NN - Deep Mind
# Playing Atari with Deep Reinforcement Learning - 2015.2
import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import time
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib

env_name = 'MountainCar-v0' #'CartPole-v1' 

env = gym.make(env_name)

#--- hyper-parameters -----
num_episodes = 300  # 'CartPole-v1' 200, 'MountainCar-v0' 300
GAMMA = 0.99
learning_rate = 0.001
hidden_layer = 120
replay_memory_size = 50_000
batch_size = 128

e_start = 0.9
e_end = 0.05
e_decay = 200

target_nn_update_frequency = 10
clip_error = False
#---------------------
device = "cpu"

n_inputs = env.observation_space.shape[0] 
n_outputs = env.action_space.n  

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.linear3 = nn.Linear(hidden_layer//2, n_outputs)

    def forward(self, x):
        a1 = torch.relu(self.linear1(x))
        a2 = torch.relu(self.linear2(a1))
        output = self.linear3(a2)
        return output

def select_action(state, steps_done):
    e_threshold = e_end + (e_start - e_end) * \
        math.exp(-1. * steps_done/e_decay)

    if random.random() > e_threshold:
        with torch.no_grad():
            state = torch.Tensor(state).to(device)
            action_values = Q(state)
            action = torch.argmax(action_values).item()
    else:
        action = env.action_space.sample()

    return action

#Initialize replay memory D to capacity N
memory = ExperienceReplay(replay_memory_size)

#Initialize action-value function Qhat with random weights theta'=theta
target_Q = NeuralNetwork().to(device)

#Initialize action-value function Q with random weights theta
Q = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

update_target_counter = 0
reward_history = []
total_steps = 0
start_time = time.time()

#for episode = 1, M do
for episode in range(num_episodes):
    if episode > num_episodes * 0.95:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
        
    #Initialize sequence s1={x1} and preprocessed sequenced f_1=f(s1)
    s, _ = env.reset()
    reward = 0
    #for t=1, T do
    while True:
        total_steps += 1

        #With probability e select a random action a_t
        #otherwise select a_t = max_a Q*(f(S_t),a;theta)
        a = select_action(s, total_steps)

        #Execute action a_t in emulator and observe reward r_t and image x_t+1
        #Set s_t+1, a_t, x_t=1 and preprocesse pi_t+1=pi(s_t+1)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        reward += r

        #Store transition(f_t, a_t, r_t, f_t+1) in D
        memory.push(s, a, s_, r, done)

        if len(memory) >= batch_size:
            #Sample random minibatch of transitions(f_j,a_j,r_j,f_j+1) from D
            states, actions, new_states, rewards, dones = memory.sample(batch_size)

            states = torch.Tensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            new_states = torch.Tensor(new_states).to(device)
            rewards = torch.Tensor([rewards]).to(device)
            dones = torch.Tensor(dones).to(device)

            new_action_values = target_Q(new_states).detach()

            #set y_j=r_j for terminal f_j+1
            # r_j + gamma*max_a'Q(f_j+1, a';theta) otherwise
            # dones가 batch_size array 이므로 dones가 1 인경우의 reward만 남김
            y_target = rewards + \
                (1 - dones) * GAMMA * torch.max(new_action_values, 1)[0]
            y_pred = Q(states).gather(1, actions.unsqueeze(1))

            #Perform a gradient descent reward on (y_j - Q(f_j+1, a_j;theta))^2
            loss = criterion(y_pred.squeeze(), y_target.squeeze())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            #Every C steps reset Qhat = Q
            if update_target_counter % target_nn_update_frequency == 0:
                target_Q.load_state_dict(Q.state_dict())

            update_target_counter += 1

        s = s_

        if done:
            reward_history.append(reward)
            print(f"{episode} episode finished after {reward:.2f} rewards")
            # scheduler.step()
            break

print("Average rewards: %.2f" % (sum(reward_history)/num_episodes))
print("Average of last 100 episodes: %.2f" % (sum(reward_history[-50:])/50))
print("---------------------- Hyper parameters --------------------------------------")
print(f"GAMMA:{GAMMA}, learning rate: {learning_rate}, hidden layer: {hidden_layer}")
print(f"replay_memory: {replay_memory_size}, batch size: {batch_size}")
print(f"epsilon_start: {e_start}, epsilon_end: {e_end}, " +
      f"epsilon_decay: {e_decay}")
print(f"update frequency: {target_nn_update_frequency}, clipping: {clip_error}")
elapsed_time = time.time() - start_time
print(f"Time Elapsed : {elapsed_time//60} min {elapsed_time%60:.0} sec")

plt.bar(torch.arange(len(reward_history)).numpy(), reward_history)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.title("DQN - Target Network")
plt.show()
