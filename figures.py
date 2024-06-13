import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotters

# Plot average regret over time in linear and logarithmic scale
regret_df = pd.read_csv("tables/07Jun241713_C-5_L-100_H-14566_N-1619_MovieLens.csv", index_col=0)


for i in regret_df:
    plotters.plot_average_regret(regret_df[i], label=i, title="Logarithmic Average Regret from MovieLens")

plt.legend(loc='upper right')
plt.yscale("log")
plt.ylabel(f"$\log (R_T / T)$")
plt.show()

for i in regret_df:
    plotters.plot_average_regret(regret_df[i], label=i, title="Average Regret from MovieLens")

plt.legend(loc='upper right')
plt.show()

# Plot the average utility over time
pred_err_df = pd.read_csv("tables/forecaster utilities for MovieLens 100.csv", index_col=0)
avg_pred_err = []
for i in pred_err_df:
    sum = 0
    avg_utility = []
    for idx, v in enumerate(pred_err_df[i]):
        sum += v
        avg_utility.append(sum/(idx + 1))
    avg_pred_err.append(avg_utility)

avg_pred_err = np.array(avg_pred_err)

names = ["random", "naive", "mfr", "recommender", "tcn"]
for i, vals in enumerate(avg_pred_err):
    if i == 5:
        continue

    plt.plot(vals, label=names[i])

plt.xlabel("Request nr.")
plt.ylabel("$U_{1:T} / T$")
plt.title("Average Utilities of Forecasters on MovieLens")
plt.legend(loc='upper right')
plt.show()

# Plot the average accuracy over time
pred_err_df = pd.read_csv("tables/forecaster accuracies for MovieLens 100.csv", index_col=0)
avg_pred_err = []
for i in pred_err_df:
    sum = 0
    avg_utility = []
    for idx, v in enumerate(pred_err_df[i]):
        sum += v
        avg_utility.append(sum/(idx + 1))
    avg_pred_err.append(avg_utility)

avg_pred_err = np.array(avg_pred_err)

names = ["random", "naive", "mfr", "recommender", "tcn"]
for i, vals in enumerate(avg_pred_err):
    if i == 5:
        continue

    plt.plot(vals, label=names[i])

plt.xlabel("Request nr.")
plt.ylabel("$A_{1:T} / T$")
plt.title("Average Accuracies of Forecasters on MovieLens")
plt.legend(loc='upper right')
plt.show()

# Plot the average Prediction Error over time
pred_err_df = pd.read_csv("tables/forecaster prediction_errs for MovieLens 100.csv", index_col=0)
avg_pred_err = []
for i in pred_err_df:
    sum = 0
    avg_utility = []
    for idx, v in enumerate(pred_err_df[i]):
        sum += v
        avg_utility.append(sum/(idx + 1))
    avg_pred_err.append(avg_utility)

avg_pred_err = np.array(avg_pred_err)

names = ["random", "naive", "mfr", "recommender", "tcn"]
for i, vals in enumerate(avg_pred_err):
    if i == 5:
        continue

    plt.plot(vals, label=names[i])

plt.xlabel("Request nr.")
plt.ylabel("$\delta_{1:T} / T$")
plt.title("Average Prediction Errors of Forecasters on MovieLens")
plt.legend(loc='upper right')
plt.show()