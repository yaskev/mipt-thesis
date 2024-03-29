from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from semipositive_network.modules.linear_bias_positive import LinearBiasPositive
from semipositive_network.modules.linear_semipositive import LinearSemiPositive
from settings import EPOCHS_COUNT, VERBOSE
from utils.batches import get_batches


class OptionsNet:
    def __init__(self, in_features: int):
        self.net = nn.Sequential(
            LinearSemiPositive(in_features, 256),
            nn.Sigmoid(),
            LinearBiasPositive(256, 1)
        )
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, analytics_mode: bool = False) -> Tuple[List[float], List[float]]:
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
        train_loss = []
        val_loss = []
        for i in range(EPOCHS_COUNT):
            if VERBOSE and i % 10 == 0 and not analytics_mode:
                print(f'Epoch: {i} out of {EPOCHS_COUNT}')
            self.net.train()
            tmp_loss = []
            for (x_batch, y_batch) in get_batches((x_train, y_train)):
                x = torch.from_numpy(x_batch)
                y = torch.from_numpy(y_batch)

                self.optim.zero_grad()
                predict = self.net.forward(x)
                loss = self.criterion.forward(predict, y.unsqueeze(1))

                l1_lambda = 3e-4
                l1_norm = sum(torch.linalg.norm(p, 1) for p in self.net.parameters())
                loss += l1_lambda * l1_norm

                loss.backward()
                self.optim.step()
                tmp_loss.append(loss.item())
            train_loss.append(sum(tmp_loss) / len(tmp_loss))

            self.net.eval()
            tmp_loss = []
            with torch.no_grad():
                for (x_batch, y_batch) in get_batches((x_val, y_val)):
                    x = torch.from_numpy(x_batch)
                    y = torch.from_numpy(y_batch)
                    predict = self.net.forward(x)
                    loss = self.criterion.forward(predict, y.unsqueeze(1))

                    l1_lambda = 3e-4
                    l1_norm = sum(torch.linalg.norm(p, 1) for p in self.net.parameters())
                    loss += l1_lambda * l1_norm

                    tmp_loss.append(loss.item())
                val_loss.append(sum(tmp_loss) / len(tmp_loss))

            self.scheduler.step()

        return train_loss, val_loss

    def predict(self, x_test: pd.DataFrame):
        self.net.eval()
        x = torch.from_numpy(x_test)
        with torch.no_grad():
            return self.net.forward(x)

    def get_greeks(self, x_greeks: pd.DataFrame) -> pd.DataFrame:
        EPS = 0.01

        x_greeks_shift = x_greeks.copy()
        x_greeks_shift[:, 0] = x_greeks_shift[:, 0] * (1 + EPS)
        x_greeks = torch.from_numpy(x_greeks)
        x_greeks_shift = torch.from_numpy(x_greeks_shift)

        predicts = []
        predicts_shift = []

        x_greeks_split = torch.split(x_greeks, 1)
        x_greeks_shift_split = torch.split(x_greeks_shift, 1)
        for i in range(len(x_greeks_split)):
            x_greeks_split[i].requires_grad = True
            predicts.append(self.net.forward(x_greeks_split[i]))

        for i in range(len(x_greeks_shift_split)):
            x_greeks_shift_split[i].requires_grad = True
            predicts_shift.append(self.net.forward(x_greeks_shift_split[i]))

        gradients = torch.autograd.grad(outputs=predicts, inputs=x_greeks_split)
        gradients = torch.cat(gradients, 0)

        gradients_shift = torch.autograd.grad(outputs=predicts_shift, inputs=x_greeks_shift_split)
        gradients_shift = torch.cat(gradients_shift, 0)

        gamma = (gradients_shift[:, 0] - gradients[:, 0]) / (EPS * x_greeks[:, 0])

        return pd.DataFrame(data={
            'delta': gradients[:, 0],
            'vega': gradients[:, 3],
            'theta': -gradients[:, 1],
            'rho': gradients[:, 2],
            'gamma': gamma
        })
