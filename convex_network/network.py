import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, Tensor

from convex_network.modules.linear_no_sum import LinearNoSum
from convex_network.modules.soft_sigmoid import SoftSigmoid
from positive_network.modules.linear_bias_positive import LinearBiasPositive
from settings import EPOCHS_COUNT, VERBOSE

from utils.batches import get_batches
from utils.plotting import create_chart


class ConvexNet:
    def __init__(self, in_features: int, convex_indices: Tensor):
        self.net = nn.Sequential(
            LinearNoSum(in_features, in_features + 3),
            SoftSigmoid(convex_indices, in_features + 3),
            LinearBiasPositive(in_features + 3, 1)
        )
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
        train_loss = []
        val_loss = []
        for i in range(EPOCHS_COUNT):
            if VERBOSE and i % 10 == 0:
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

        create_chart(train_loss, val_loss, 'convex_network')

    def predict(self, x_test: pd.DataFrame):
        self.net.eval()
        x = torch.from_numpy(x_test)
        with torch.no_grad():
            return self.net.forward(x)
