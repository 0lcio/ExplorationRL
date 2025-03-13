import json
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from base_model import netCounterBase
from constants import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES_IG, NUM_CLASSES_BASE
from ig_model import NetIG


def plot_metrics(data, xlabel, ylabel, data_std=None, title=None, legend_labels=None, save_path=None, marker="o", max_y=None):
    plt.figure(figsize=(10, 6))
    for i, values in enumerate(data):
        values = np.array(values)
        if data_std:
            std_dev = np.array(data_std[i]) if data_std else np.zeros_like(values)
        
        if legend_labels and legend_labels[i] == "Random agent":
            plt.plot(values, label=legend_labels[i], marker=marker, linestyle='--', color='red')
        else:
            plt.plot(values, label=legend_labels[i] if legend_labels else None, marker=marker)
        
        if data_std:
            plt.fill_between(range(len(values)), values - std_dev, values + std_dev, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if max_y is not None:
        plt.ylim(top=max_y)
    if title:
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
    """Calcola l'entropia di un tensore di probabilità."""
    # Utilizza mascheramento per evitare log(0) e moltiplicazioni inutili
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(prediction):
    """Calcola l'information gain dato un'osservazione come tensore."""
    prob_uni = np.full(8, 1 / 8)  # Distribuzione uniforme iniziale
    predictions = torch.softmax(prediction, dim=1).detach().cpu().numpy()
    ig = entropy(prob_uni) - entropy(predictions)
    return ig


def load_models():
    base_model = netCounterBase(NUM_CLASSES_BASE, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    ig_model = NetIG(NUM_CLASSES_IG, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    base_model_path = os.path.join('data', 'fold_1_saved_base_model.pth')
    ig_model_path = os.path.join('data', 'fold_1_saved_ig_model_global_MSE_entropy_loss_head.pth')
    if torch.cuda.is_available():
        base_model.load_state_dict(torch.load(base_model_path))
        ig_model.load_state_dict(torch.load(ig_model_path))
    else:
        base_model.load_state_dict(torch.load(base_model_path, map_location=torch.device('cpu'), weights_only=True))
        base_model.load_state_dict(torch.load(base_model_path, map_location=torch.device('cpu'), weights_only=True))

    return base_model, ig_model
