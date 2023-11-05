from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from positive_network.modules.linear_bias_positive import LinearBiasPositive
from positive_network.modules.linear_positive import LinearPositive
from settings import EPOCHS_COUNT, VERBOSE
from utils.batches import get_batches


class SigmaNetType(Enum):
    POS_NET_LIKE = 'pos_net_like'
    POS_NET_WIDE = 'pos_net_wide'
    MULTILAYER_FFNN = 'multilayer_ffnn'
    SMALL_FFNN = 'small_ffnn'
    MULTILAYER_POSITIVE = 'multilayer_positive'


class SigmaNet:
    def __init__(self, in_features: int, net_type: SigmaNetType):
        if net_type == SigmaNetType.MULTILAYER_FFNN:
            self.net = nn.Sequential(
                nn.Linear(in_features, in_features + 128),
                nn.Sigmoid(),
                nn.Linear(in_features + 128, in_features + 64),
                nn.Sigmoid(),
                nn.Linear(in_features + 64, in_features + 32),
                nn.Sigmoid(),
                nn.Linear(in_features + 32, in_features + 16),
                nn.Sigmoid(),
                nn.Linear(in_features + 16, in_features + 8),
                nn.Sigmoid(),
                nn.Linear(in_features + 8, 1)
            )
        elif net_type == SigmaNetType.MULTILAYER_POSITIVE:
            self.net = nn.Sequential(
                nn.Linear(in_features, in_features + 128),
                nn.Sigmoid(),
                nn.Linear(in_features + 128, in_features + 64),
                nn.Sigmoid(),
                nn.Linear(in_features + 64, in_features + 32),
                nn.Sigmoid(),
                nn.Linear(in_features + 32, in_features + 16),
                nn.Sigmoid(),
                nn.Linear(in_features + 16, in_features + 8),
                nn.Sigmoid(),
                LinearBiasPositive(in_features + 8, 1)
            )
        elif net_type == SigmaNetType.POS_NET_LIKE:
            self.net = nn.Sequential(
                LinearPositive(in_features, in_features + 3),
                nn.Sigmoid(),
                LinearBiasPositive(in_features + 3, 1)
            )
        elif net_type == SigmaNetType.POS_NET_WIDE:
            self.net = nn.Sequential(
                LinearPositive(in_features, in_features * 8),
                nn.Sigmoid(),
                LinearBiasPositive(in_features * 8, 1)
            )
        elif net_type == SigmaNetType.SMALL_FFNN:
            self.net = nn.Sequential(
                nn.Linear(in_features, in_features * 8),
                nn.Sigmoid(),
                nn.Linear(in_features * 8, 1)
            )
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

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
                    tmp_loss.append(loss.item())
                val_loss.append(sum(tmp_loss) / len(tmp_loss))

        return train_loss, val_loss

    def predict(self, x_test: pd.DataFrame):
        self.net.eval()
        x = torch.from_numpy(x_test)
        with torch.no_grad():
            return self.net.forward(x)
