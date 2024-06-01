import numpy as np
import random

# Example titles for an article
titles = ["Title A", "Title B", "Title C"]

Sim = 100

# Thompson Sampling Algorithm
def thompson_sampling(victory_chance, T, seed=2016):
    J = len(titles)

    S = np.zeros(J, dtype=int)
    F = np.zeros(J, dtype=int)

    rnd = np.random.RandomState(seed)

    machines  = np.zeros((T, J))
    E_victory_p = np.zeros((T, J))
    V_victory_p = np.zeros((T, J))

    for t in range(T):
        # Drawing from each arm
        victory_chance_t = rnd.beta(S + 1, F + 1) 
        # Choosing the best one in that particular round
        machine = np.argmax(victory_chance_t)      
        # Simulating click (success) or no click (failure)
        success_t = rnd.binomial(1, victory_chance[machine]) 
        # Updating success and failures
        S[machine] += success_t                    
        F[machine] += 1 - success_t                

        machines[t, :] = np.eye(1, J, k=machine)

        # Updating and calculating mean and variance for chosen arm
        E_victory_p[t, :] = (S + 1) / (S + F + 2)
        V_victory_p[t, :] = (S + 1) * (F + 1) / (((S + F + 2) ** 2) * (S + F + 3))

        total_victories = np.sum(S)
    
    return [total_victories, machines, E_victory_p, V_victory_p, S, F]

total_victories_thompson = []
