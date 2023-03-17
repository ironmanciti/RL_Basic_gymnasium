# Deep Q-learning Target NN and DDQN - Deep Mind
# Playing Atari with Deep Reinforcement Learning - 2015.2
import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

env_name = 'MountainCar-v0' #'LunarLander-v2', 'CartPole-v1'

env = gym.make(env_name)

seed_val = 23
torch.manual_seed(seed_val)
random.seed(seed_val)
#--- hyper-parameters -----
num_episodes = 500
gamma = 0.99
learning_rate = 0.01  #faster or stable training
hidden_layer = 64
replay_memory_size = 50000
batch_size = 32

target_nn_update_frequency = 500

double_dqn = True  #False - original DQN
if double_dqn:
    print("Double DQN ...")

egreedy = 1.0
egreedy_final = 0.1
egreedy_decay = 500

clip_error = True

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"device = {device}")

n_inputs = env.observation_space.shape[0]  # 4
n_outputs = env.action_space.n  # 2

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    return epsilon

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
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, n_outputs)

    def forward(self, x):
        a1 = torch.tanh(self.linear1(x))
        output = self.linear2(a1)
        return output


def select_action(state, epsilon):
    if torch.rand(1).item() > epsilon:
        with torch.no_grad():
            state = torch.Tensor(state).to(device)
            action_values = Q(state)
            action = torch.argmax(action_values).item()
    else:
        action = env.action_space.sample()

    return action

#Initialize replay memory D to capacity N
memory = ExperienceReplay(replay_memory_size)

#Initialize action-value function Q with random weights theta
Q = NeuralNetwork().to(device)
#Initialize action-value function Qhat with random weights theta'=theta
target_Q = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

update_target_counter = 0
steps_history = []
total_steps = 0
#for episode = 1, M do
for i_episode in range(num_episodes):
    if i_episode < num_episodes * 0.05 or i_episode > num_episodes * 0.95:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    #Initialize sequence s1={x1} and preprocessed sequenced f_1=f(s1)
    s, _ = env.reset()
    step = 0
    #for t=1, T do
    while True:
        step += 1
        total_steps += 1

        e = calculate_epsilon(total_steps)
        #With probability e select a random action a_t
        #otherwise select a_t = max_a Q*(f(S_t),a;theta)
        a = select_action(s, e)

        #Execute action a_t in emulator and observe reward r_t and image x_t+1
        #Set s_t+1, a_t, x_t=1 and preprocesse pi_t+1=pi(s_t+1)
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        #Store transition(f_t, a_t, r_t, f_t+1) in D
        memory.push(s, a, s_, r, done)

        if len(memory) >= batch_size:
            #Sample random minibatch of transitions(f_j,a_j,r_j,f_j+1) from D
            states, actions, new_states, rewards, dones = memory.sample(
                batch_size)

            states = torch.Tensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            new_states = torch.Tensor(new_states).to(device)
            rewards = torch.Tensor([rewards]).to(device)
            dones = torch.Tensor(dones).to(device)

            if double_dqn:
                new_values = Q(new_states).detach()
                max_action_indexes = torch.max(new_values, 1)[1]
                new_action_values = target_Q(new_states).detach()
                max_new_state_values = \
                    new_action_values.gather(1, max_action_indexes.unsqueeze(1)).squeeze(1)
            else:
                new_action_values = target_Q(new_states).detach()
                max_new_state_values = torch.max(new_action_values, 1)[0]

            #set y_j=r_j for terminal f_j+1
            # r_j + gamma*max_a'Q(f_j+1, a';theta) otherwise
            # dones가 batch_size array 이므로 dones가 1 인경우의 reward만 남김
            y_target = rewards + (1 - dones) * gamma * max_new_state_values
            y_pred = Q(states).gather(1, actions.unsqueeze(1))

            #Perform a gradient descent step on (y_j - Q(f_j+1, a_j;theta))^2
            loss = criterion(y_pred.squeeze(), y_target.squeeze())
            optimizer.zero_grad()
            loss.backward()

            if clip_error:
                for param in Q.parameters():
                    param.grad.data.clamp_(-1, 1)

            optimizer.step()

            #Every C steps reset Qhat = Q
            if update_target_counter % target_nn_update_frequency == 0:
                target_Q.load_state_dict(Q.state_dict())

            update_target_counter += 1

        s = s_

        if done:
            steps_history.append(step)
            print(f"{i_episode} episode finished after {step} steps")
            break

print("Average steps: %.2f" % (sum(steps_history)/num_episodes))
print("Average of last 100 spisodes: %.2f" % (sum(steps_history[-50:])/50))
plt.bar(torch.arange(len(steps_history)).numpy(), steps_history)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.show()
