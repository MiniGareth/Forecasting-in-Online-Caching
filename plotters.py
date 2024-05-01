import matplotlib.pyplot as plt
import numpy as np


def plot_regret(regret_list, title=None):
    plt.figure()
    if title is None:
        plt.title("Regret per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$R_T$")
    plt.xticks(np.arange(len(regret_list), step=len(regret_list)//10))
    plt.plot(regret_list)
    plt.show()

def plot_regret_over_time(regret_list, title=None):
    plt.figure()
    if title is None:
        plt.title("Regret over Time per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$R_T / T$")
    plt.xticks(np.arange(len(regret_list), step=len(regret_list)//10))
    plt.plot([regret / i for i, regret in enumerate(regret_list)])
    plt.show()