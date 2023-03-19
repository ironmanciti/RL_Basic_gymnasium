# Q-Learning (off-policy TD control) for estimating pi=pi*
# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)


#Algorithm parameters: step size alpha, small e>0


# Initialize Q(s,a) arbitrarily except that Q(terminal, .)=0


# Loop for each episode:
for episode in range(n_episodes):
    # if episode > n_episodes * 0.995:
    #     env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode="human")
    # Initialize S
    
    # Loop for each step of episode:
    
    while True:
        # Choose A from S using policy derived from Q (eg. e-greedy)
        # behavior policy : e-greedy
        

        # Take action A, observe R, S'
        
        #Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        #자신이 따르는 정책에 상관없이 최적 행동가치함수 q*를 직접 근사
        # target policy : greedy policy
        

        # S <-- S'
        
        # until S is terminal

   


#optimal policy 출력
