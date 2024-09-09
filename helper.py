import json

import matplotlib.pyplot as plt


def plot_metrics(data, xlabel, ylabel, title, legend_labels=None, save_path=None):
    plt.figure(figsize=(10, 6))
    for i, values in enumerate(data):
        plt.plot(values, label=legend_labels[i] if legend_labels else None, marker="o")
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
