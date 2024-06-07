import numpy as np
import torch
from torch.autograd import Variable

from forecasters.Forecaster import Forecaster
from tcn.models import TemporalConvNet


# Hyper parameters found to be best for MovieLens
default_params = {
        'num_inputs': 100,
        'num_filters': 40,
        'num_layers': 8,
        'kernel_size': 6,
        'dropout': 0.2,
        'num_classes': 100,
        'learning_rate': 0.1
    }
class TCNForecaster(Forecaster):
    def __init__(self, num_inputs=None, num_filters=None, num_layers=None, kernel_size=None, dropout=None, num_classes=None,
                 gpu=True, model_path=None, history=None, horizon=128):
        super().__init__(horizon)
        self.model = TemporalConvNet(
            num_inputs=num_inputs if num_inputs is not None else default_params['num_inputs'],
            num_channels=[num_filters if num_filters is not None else default_params['num_filters']] * (num_layers if num_layers is not None else default_params['num_layers']),
            kernel_size=kernel_size if kernel_size is not None else default_params['kernel_size'],
            dropout=dropout if dropout is not None else default_params['dropout'],
            runs_folder="runs",
            mode="classification",
            num_classes=num_classes if num_classes is not None else default_params['num_classes'],
            gpu=gpu
        )
        if gpu is True:
            self.model.cuda()

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        if history is None:
            self.history = []
        if history is not None and model_path is None:
            # TODO train on the given history if given model path is None
            pass
        else:
            self.history = list(history)

    def predict(self) -> np.ndarray:
        input = np.array(self.history[-self.horizon:]).T
        input = np.array([input])
        input = Variable(torch.from_numpy(input)).float()
        if self.model.gpu:
            input = input.cuda()

        output = self.model(input)
        prediction = torch.exp(output)
        prediction = prediction.cpu().detach().numpy()
        return prediction[0]

    def update(self, latest_req:np.ndarray):
        self.history.append(latest_req)