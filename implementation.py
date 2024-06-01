from ts_script import thompson_sampling
import json
import requests
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import beta

# Skeleton functions, replace with actual names of variables, and actual api of the website
def get_data():
    # Example url, replace with actual
    reactions = requests.get("https://your-cms.com/api/update-title")
    data = reactions.json
    return data['clicks'], data['impressions'], data['titles']

def get_num_success(titles, clicks, impressions):
    S = np.array(clicks)
    F = np.array(impressions) - S
    return S, F

def main():
    titles, clicks, impressions = get_data()
    S, F = get_num_success(titles, clicks, impressions)
    victory_chance = S / (S + F)
    T = len(impressions)

    total_victories, machines, E_victory_p, V_victory_p, S, F = thompson_sampling(victory_chance, T)

    best_title_index = np.argmax(S)
    best_title = titles[best_title_index]
    
    # Update the displayed title on your website
    requests.post('https://your-cms.com/api/update-title', json={'title': best_title})

if __name__ == "__main__":
    main()