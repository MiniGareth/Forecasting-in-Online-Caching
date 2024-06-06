import numpy as np
import torch

from forecasters.Forecaster import Forecaster
from tcn.models import TemporalConvNet

default_params = {
        "num_inputs": [100],
        "num_filters": [40, 50, 60],
        "num_layers": [6, 8, 10, 12],
        "kernel_size": [6, 8],
        "dropout": [0.2],
        "num_classes": [100],
        "learning_rate": [0.1],
    }
class TCNForecaster(Forecaster):
    def __init__(self, num_inputs, num_filters, num_layers, horizon=128, kernel_size=2, dropout=0.2, num_classes=1,
                 gpu=True, model_path=None, history: list=None):
        super().__init__(horizon)
        if history is None:
            history = []
        self.model = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=[num_filters] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            runs_folder="runs",
            mode="classification",
            num_classes=num_classes,
            gpu=gpu
        )

        if gpu is True:
            self.model.cuda()

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        self.history = history

    def predict(self) -> np.ndarray:
        input = self.history[-self.horizon:, :].T
        prediction = np.exp(torch.asarray(self.model(input)))

        return prediction

    def update(self, history_vec: list):
        self.history = history_vec