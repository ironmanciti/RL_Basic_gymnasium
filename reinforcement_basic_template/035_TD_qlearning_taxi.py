# Q-Learning (off-policy TD control) for estimating pi=pi*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os
"""
6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
          5 * 5 * 5 * 4 = 500

Rewards:
    per-step : -1,
    delivering the passenger : +20,
    executing "pickup" and "drop-off" actions illegally : -10
    
blue: passenger
magenta: destination
yellow: empty taxi
green: full taxi
"""


#Algorithm parameter: step size alpha (0,1], small e > 0

#Initialize Q(s,a) for all s, a arbitrarily except Q(terminal,.)=0


#Loop for each episode:
for episode in range(n_episodes):
    if episode > n_episodes * 0.995:
        env = gym.make('Taxi-v3', render_mode="human")
    #Initialize S
    
    #Loop for each step of episode:
    while True:
        step += 1
        # Choose A from S using policy derived from Q (eg. e-greedy)
        # behavior policy : e-greedy
        

        #Take action A, observe R, S'
        

        #Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        #자신이 따르는 정책에 상관없이 최적 행동가치함수 q*를 직접 근사
        # target policy : greedy policy
        

        #S <- S'
        
    
