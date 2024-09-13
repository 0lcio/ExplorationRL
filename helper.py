import json
import math

import matplotlib.pyplot as plt


def plot_metrics(data, xlabel, ylabel, title, legend_labels=None, save_path=None, marker="o"):
    plt.figure(figsize=(10, 6))
    for i, values in enumerate(data):
        plt.plot(values, label=legend_labels[i] if legend_labels else None, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if legend_labels:
        plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_dict(dictionary, file_path):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def read_dict(file_path):
    with open(file_path, "r") as file:
        load_dictionary = json.load(file)

    return load_dictionary


def entropy(probabilities):
    """Calcola l'entropia di una lista di probabilità."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def information_gain(obs):
    """Calcola l'information gain dato un'osservazione."""
    prob_uni = [1 / 8] * 8  # Distribuzione uniforme iniziale
    return entropy(prob_uni) - entropy(obs)
