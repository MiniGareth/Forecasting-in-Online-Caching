from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotters

root_dir = Path(".").resolve()

# Plot average regret over time in linear and logarithmic scale
regret_df1 = pd.read_csv(str(root_dir / "tables" / "19Jun241532_C-5_L-100_H-14566_N-1619_MovieLens.csv"), index_col=0)
regret_df2 = pd.read_csv(str(root_dir / "tables" / "19Jun241532_C-5_L-100_H-14566_N-1619_MovieLens.csv"), index_col=0)
regret_df3 = pd.read_csv(str(root_dir / "tables" / "19Jun241532_C-5_L-100_H-14566_N-1619_MovieLens.csv"), index_col=0)
for i, regret_df in enumerate([regret_df3]):
    for label in regret_df:
        if label == "parrot":
            continue
        if label == "zero":
            final_val = np.array(regret_df[label])[-1] / len(regret_df)
            plt.plot([final_val for j in range(len(regret_df[label]))], label=label)
            continue

        plotters.plot_average_regret(regret_df[label], label=label, title=f"Logarithmic Average Regret from MovieLens")

    plt.legend(loc='upper right')
    plt.yscale("log")
    plt.ylabel(f"$\log (R_T / T)$")
    plt.show()
for i, regret_df in enumerate([regret_df3]):
    for label in regret_df:
        if label == "parrot" or label == "random" or label == "naive" or label == "mfr" or label == "zero":
            continue
        plotters.plot_average_regret(regret_df[label], label=label, title=f"Logarithmic Average Regret Probability vs One-Hot Vector")

    plt.legend(loc='upper right')
    plt.yscale("log")
    plt.ylabel(f"$\log (R_T / T)$")
    plt.show()

def plot_avg_forecaster_stats(df):
    avg_stat = []
    for i in df:
        print(i)
        sum = 0
        avg_utility = []
        for idx, v in enumerate(df[i]):
            sum += v
            avg_utility.append(sum/(idx + 1))
        avg_stat.append(avg_utility)

    avg_stat = np.array(avg_stat)

    for label, vals in zip(df, avg_stat):
        print(label)
        if label == "popularity" or label == "popularity one-hot" or label == "des" or label == "des on-hot" or label == "Parrot 50":
            continue
        plt.plot(vals, label=label)

    plt.xlabel("Request nr.")
    plt.legend(loc='upper right')

# Plot the average utility over time
utilities_df = pd.read_csv(str(root_dir / "tables" / "forecaster utilities for MovieLens 100.csv"), index_col=0)
plot_avg_forecaster_stats(utilities_df)
plt.ylabel("$U_{1:T} / T$")
plt.title("Average Utilities of Forecasters on MovieLens")
plt.show()

# Plot the average accuracy over time
acc_df = pd.read_csv(str(root_dir / "tables" / "forecaster accuracies for MovieLens 100.csv"), index_col=0)
plot_avg_forecaster_stats(acc_df)
plt.ylabel("$A_{1:T} / T$")
plt.title("Average Accuracies of Forecasters on MovieLens")
plt.show()

# Plot the average Prediction Error over time
pred_err_df = pd.read_csv(str(root_dir / "tables" / "forecaster prediction_errs for MovieLens 100.csv"), index_col=0)
plot_avg_forecaster_stats(pred_err_df)
plt.ylabel("$\delta_{1:T} / T$")
plt.title("Average Prediction Errors of Forecasters on MovieLens")
plt.legend(loc="center right")
plt.show()