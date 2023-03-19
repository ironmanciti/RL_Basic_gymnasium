# Suntton p.101
# On-Policy First-Visit MC control(for e-soft policies) for optimum policy pi*


#state : (player card sum, dealer open card, usable_ace 보유) ex) (6, 1, False)


#Algorithm parameter: small e > 0




#Initialize
#pi <- an arbitrary e-soft policy (초기 policy)
#Q(s,a) 초기화 for all s, a
#Returns(s, a) <- empty list for all s, a


#Repeat forever (for each episode)
for i in range(num_episodes):
    #Generate an episode following pi: S0,A0,R1,S1,A1,R2,..ST-1,AT-1,RT
    
    while True:
        
        #s:(sum_hand(player), dealer open card, usable_ace 보유)
        

    #G <- 0
    
    #Loop for each step of episode, t=T-1, T-2,...0
    for s, a, r in episode[::-1]:
        # G <- gamma*G + R_(t+1)
        
        #Unless the pair S_t, A_t appears in S_0,A_0 S_1,A_1..S_(t-1),A_(t-1):
        #Append G to Returns(S_t, A_t)
        #Q(S_t,A_t) <- average(Returns(S_t, A_t))
        
        #A* <- argmax_a Q(S_t,a)
        
        #For all a:
        #pi(a|S_t) <- 1-e + e/|A(S_t)| if a = A*
        #          <- e/|A(St)|        if a != A*
        

    

# prediction
