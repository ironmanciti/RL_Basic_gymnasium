import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

# Simple deterministic Policy
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

stochastic = False
with_policy = True
mode = None
if stochastic: # stochastic environment
    env = gym.make('FrozenLake-v1',
                   desc=None,
                   map_name="4x4",
                   is_slippery=True)
else:  # deterministic environment
    env = gym.make('FrozenLake-v1',
                   desc=None,
                   map_name="4x4",
                   is_slippery=False)

n_games = 100
win_pct = []
scores = []

for i in range(n_games):
    terminated, truncated = False, False
    obs, info = env.reset()
    score = 0
    while not terminated and not truncated:
          
        if with_policy:  # simple deterministic polity
            action = policy[obs]
        else:  # No policy
            action = env.action_space.sample() 
            
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

    scores.append(score)

    if i % 10:
        average = np.mean(scores[-10:])
        win_pct.append(average)

env.close()

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('With Policy: average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if stochastic else 'Deterministic Env'))
plt.show()
