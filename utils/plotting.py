import os
from datetime import datetime
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt


def create_chart(train_loss: List[float], val_loss: List[float], folder: str, strict_path: str = None):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(train_loss)), train_loss, label='Train')
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.grid()
    plt.yscale('log')
    plt.title('Loss', fontsize=18)
    plt.legend()

    if strict_path is not None:
        plt.savefig(strict_path)
    else:
        plt.savefig(os.path.join(folder, 'charts', f'{datetime.now()}.jpg'))


def plot_different_metrics(metrics: Dict[str, List[float]], dataset_sizes: List[float], strict_path: str, metric_name: str):
    plt.figure(figsize=(12, 8))

    for name, metric in metrics.items():
        plt.plot(dataset_sizes, metric, label=name)

    plt.xlabel('Dataset size')
    plt.ylabel(metric_name.capitalize())
    plt.yscale('log')
    plt.grid()
    plt.title(metric_name.capitalize(), fontsize=18)
    plt.legend()
    plt.savefig(strict_path)
