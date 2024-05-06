import matplotlib.pyplot as plt
import numpy as np


def plot_cummulative_regret(regret_list, title=None):
    plt.figure()
    if title is None:
        plt.title("Cummulative Regret per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$R_T$")
    plt.xticks(np.arange(len(regret_list), step=len(regret_list)//10))
    cummulative_regret = []
    for i, regret in enumerate(regret_list):
        if i == 0:
            cummulative_regret.append(regret)
        else:
            cummulative_regret.append(cummulative_regret[i - 1] + regret)
    plt.plot(cummulative_regret)
    plt.show()

def plot_average_regret(regret_list, title=None):
    plt.figure()
    if title is None:
        plt.title("Average Regret per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$R_T / T$")
    plt.xticks(np.arange(len(regret_list), step=len(regret_list)//10))
    plt.plot([regret / i for i, regret in enumerate(regret_list)])
    plt.show()