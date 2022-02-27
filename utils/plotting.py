import os
from datetime import datetime
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def create_chart(train_loss: List[float], val_loss: List[float], folder: str):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(train_loss)), train_loss, label='Train')
    plt.plot(np.arange(len(val_loss)), val_loss, label='Validation')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.grid()
    plt.yscale('log')
    plt.title('Loss', fontsize=18)
    plt.legend()
    plt.savefig(os.path.join(folder, 'charts', f'{datetime.now()}.jpg'))
