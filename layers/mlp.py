from torch import nn

from layers.dense import Dense
from layers import utils


class MLP(nn.Module):
    """
    Simple Multi-layer Perceptron encoder with optional linear readout.
    """

    def __init__(self, input_size, hidden_size, output_size=None, n_layers=1, dropout=0.0):
        super().__init__()

        layers = [
            Dense(
                input_size=input_size if i == 0 else hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
            for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter("readout", None)

    def forward(self, x, u=None):
        """"""
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out