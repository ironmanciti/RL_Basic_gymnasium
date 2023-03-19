# Suntton p.92 First-Visit MC predictions, for estimating V ~ v_pi
# card 조합에 따른 State Value estimate



# Input: a policy pi to be evaluated


def pi(state):
    # state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False)
    # player card 가 stick_threshold 이상이면 무조건 stick 하고
    # else 이면 hit 하는 전략
    #0:stick, 1:hit


# Initialize V(s)
# Returns(s) <- empty list for all s


# Loop forever(for each episode)
for i in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    
    
    #G <- 0
    
    #Loop for each step of spisode: t=T-1,T-2,...,0
    for s, a, r in episode[::-1]:
        #G <- gamma*G + R_(t+1)
        
        #Unless S_t appears in S_0, S_1,...S_(t-1):
        if s not in visited_states:
            #Append G to Returns(S_t)
            
            #V(S_t) <- average(Returns(S_t))
            


