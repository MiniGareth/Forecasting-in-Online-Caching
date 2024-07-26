import matplotlib.pyplot as plt
import numpy as np

def plot_cummulative_regret(regret_list, title=None):
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
    # plt.show()

def plot_average_regret(regret_list, title=None, label=None):
    if title is None:
        plt.title("Average Regret per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$R_T / T$")
    plt.xticks(np.arange(len(regret_list), step=len(regret_list)//10))
    plt.plot([regret / (i + 1) for i, regret in enumerate(regret_list)], label=label, linewidth=1)
    # plt.show()

def plot_average_utility(utility_list, title=None, label=None):
    if title is None:
        plt.title("Utility per request")
    else:
        plt.title(title)
    plt.xlabel("Request nr.")
    plt.ylabel(f"$U_T / T$")
    plt.xticks(np.arange(len(utility_list), step=len(utility_list) // 10))
    plt.plot([utility / (i + 1) for i, utility in enumerate(utility_list)], label=label)

def plot_utility_bar(data: list, forecaster_list: list, distributions: list, title=None):
    if len(data) != len(distributions):
        raise ValueError("The number of distributions must match the number of utility lists in data")
    if title is None:
        plt.title("Utility per Forecaster")
    else:
        plt.title(title)

    X_axis = np.arange(len(forecaster_list))

    width = 0.8 / len(distributions)
    loc = -1 * width * (len(distributions) - 1) / 2
    for i in range(len(distributions)):
        plt.bar(X_axis + loc, data[i], width, label=distributions[i])
        loc += width

    plt.xlabel("Forecasters")
    plt.ylabel("Utility")
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.xticks(X_axis, forecaster_list)
    plt.legend(loc="upper right")

