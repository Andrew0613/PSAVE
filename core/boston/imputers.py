import numpy as np
import warnings
import torch


class MarginalImputer():
    '''Marginalizing out removed features with their marginal distribution.'''
    def __init__(self, model, data):
        # super().__init__(model)
        device = next(model.parameters()).device
        self.model = lambda x: model(torch.tensor(
            x, dtype=torch.float32, device=device)).cpu().data.numpy()
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = data.shape[1]

        if len(data) > 1024:
            warnings.warn('using {} background samples may lead to slow '
                          'runtime, consider using <= 1024'.format(
                            len(data)), RuntimeWarning)

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)

        x = np.tile(x, (self.samples, 1))  # x.repeat(self.samples, 0)
        S = np.tile(S, (self.samples * n, 1))  # S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        x_ = torch.Tensor(x_)
        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)
