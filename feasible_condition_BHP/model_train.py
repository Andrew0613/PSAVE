import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import net


feature_num = 10
save_path = "./model/test_net_10.pt"
torch.set_default_tensor_type(torch.DoubleTensor)

# read data
dataset = pd.read_csv("./data/boston.csv")

# prepare model
model = net.TestNet(feature_num, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
epochs = 100
Xs = dataset.iloc[:, 0:feature_num]
Ys = dataset.iloc[:, 13]
Xs = torch.Tensor(np.array(Xs))
Ys = torch.Tensor(np.array(Ys))

for epoch in range(epochs):
    var_x = Variable(Xs, requires_grad=True)
    var_y = Variable(Ys, requires_grad=True)

    optimizer.zero_grad()
    out = model(var_x)
    loss = criterion(out, var_y)

    loss.backward()
    optimizer.step()

    print(loss)

torch.save(model.state_dict(), save_path)
