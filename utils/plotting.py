import os
from datetime import datetime
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from settings import NUMBER_OF_STEPS


def create_chart(train_loss: List[float], val_loss: List[float], folder: str, tag: str = '', strict_path: str = None):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation')
    plt.plot(np.arange(len(train_loss)), train_loss, label='Train')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.grid()
    plt.yscale('log')
    plt.title('Loss', fontsize=18)
    plt.legend()

    if strict_path is not None:
        plt.savefig(f'{tag}{strict_path}')
    else:
        plt.savefig(os.path.join(folder, 'charts', f'{tag}{datetime.now()}.jpg'))


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


def plot_loss_comparison(loss_cmp: Dict[int, Dict[str, List[float]]], strict_path: str):
    for dataset_size, loss_dict in loss_cmp.items():
        plt.figure(figsize=(8, 6))

        for name, loss in loss_dict.items():
            plt.plot(np.arange(len(loss)), loss, label=name)

        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid()
        plt.title(f'Loss, dataset size={dataset_size}', fontsize=18)
        plt.legend()
        plt.savefig(os.path.join(strict_path, f'{dataset_size}.jpg'))


def plot_many_paths(paths: np.ndarray, strict_path: str):
    plt.figure(figsize=(8, 6))

    for one_path in paths:
        plt.plot(np.arange(NUMBER_OF_STEPS + 1), one_path)

    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.yscale('log')
    plt.grid()
    # plt.title('MPaths, scale=log', fontsize=18)
    plt.savefig(os.path.join(strict_path, f'{datetime.now()}.svg'))
