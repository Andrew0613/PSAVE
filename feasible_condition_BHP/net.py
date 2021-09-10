import torch
import torch.nn as nn
import torch.nn.functional as func


class TestNet(nn.Module):
    def __init__(self, input_num, output_num):
        super(TestNet, self).__init__()
        self.layer_1 = nn.Linear(input_num, 20)
        self.layer_2 = nn.Linear(20, 20)
        self.layer_3 = nn.Linear(20, output_num)

    def forward(self, x):
        x = func.sigmoid(self.layer_1(x))
        x = func.sigmoid(self.layer_2(x))
        x = self.layer_3(x)
        return x
