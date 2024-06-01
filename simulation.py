from ts_script import thompson_sampling
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta

# Number of trials
T = 100  # Number of rounds per simulation
S = 100
titles = ["Title A", "Title B", "Title C"]
total_victories_thompson = []

for s in range(S):
    seed = 2016 + s  # Change seed for each simulation
    victory_chance = [0.2, 0.35, 0.5]  # Example click through rates for each title
    [total_victories, machines, E_victory_p, V_victory_p, S, F] = thompson_sampling(victory_chance, T, seed)
    total_victories_thompson.append(total_victories)

average_total_victories_thompson = np.mean(total_victories_thompson)

print(f"Average total victories over {S} simulations: {average_total_victories_thompson}")
print(f"Final successes: {S}")
print(f"Final failures: {F}")
print(f"Estimated victory probabilities (mean): \n{E_victory_p[-1, :]}")
print(f"Victory probabilities variances: \n{V_victory_p[-1, :]}")

x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(titles)):
    y = beta.pdf(x, S[i] + 1, F[i] + 1)
    ax.plot(x, y, label=f"{titles[i]}: a={S[i] + 1}, b={F[i] + 1}")

ax.set_title("Posterior PDFs of Victory Probabilities")
ax.set_xlabel("Victory Probability")
ax.set_ylabel("Density")
ax.legend(title="Titles")
plt.show()
