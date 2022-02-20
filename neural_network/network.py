from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from settings import EPOCHS_COUNT
from .modules.linear_positive import LinearPositive
from .modules.linear_bias_positive import LinearBiasPositive
import matplotlib.pyplot as plt


class OptionsNet:
    def __init__(self, in_features: int):
        self.net = nn.Sequential(
            LinearPositive(in_features, in_features + 3),
            nn.Sigmoid(),
            LinearBiasPositive(in_features + 3, 1)
        )
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
        train_loss = []
        val_loss = []
        for _ in range(EPOCHS_COUNT):
            self.net.train()
            tmp_loss = []
            for (x_batch, y_batch) in OptionsNet.get_batches((x_train, y_train)):
                x = torch.from_numpy(x_batch)
                y = torch.from_numpy(y_batch)

                self.optim.zero_grad()
                predict = self.net.forward(x)
                loss = self.criterion.forward(predict, y.unsqueeze(1))
                loss.backward()
                self.optim.step()
                tmp_loss.append(loss.item())
            train_loss.append(sum(tmp_loss) / len(tmp_loss))

            self.net.eval()
            tmp_loss = []
            with torch.no_grad():
                for (x_batch, y_batch) in OptionsNet.get_batches((x_val, y_val)):
                    x = torch.from_numpy(x_batch)
                    y = torch.from_numpy(y_batch)
                    predict = self.net.forward(x)
                    loss = self.criterion.forward(predict, y.unsqueeze(1))
                    tmp_loss.append(loss.item())
                val_loss.append(sum(tmp_loss) / len(tmp_loss))

        OptionsNet.create_chart(train_loss, val_loss)

    def predict(self, x_test: pd.DataFrame):
        self.net.eval()
        x = torch.from_numpy(x_test)
        with torch.no_grad():
            return self.net.forward(x)

    @staticmethod
    def get_batches(dataset, batch_size=100):
        X, Y = dataset
        n_samples = X.shape[0]

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            yield X[batch_idx], Y[batch_idx]

    @staticmethod
    def create_chart(train_loss, val_loss):
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(train_loss)), train_loss, label='Train')
        plt.plot(np.arange(len(val_loss)), val_loss, label='Validation')
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.grid()
        plt.yscale('log')
        plt.title('Loss', fontsize=18)
        plt.legend()
        plt.savefig(f'charts/{datetime.now()}.jpg')
