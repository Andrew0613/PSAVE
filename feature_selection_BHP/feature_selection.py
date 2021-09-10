import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import net


feature_num = 3
save_path = "./model/feature_selection.pt"
torch.set_default_tensor_type(torch.DoubleTensor)

# read data
dataset = pd.read_csv("./data/boston.csv")

# prepare model
model = net.TestNet(feature_num, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
epochs = 35
X = np.array(dataset.iloc[:, 0:10])
Ys = np.array(dataset.iloc[:, 13])
l1 = [8,1,2,5,4,7,3,6,0]#fsave
l2 = [2,6,0,3,4,5,1,7,8]#sage
result = [[] for i in range (10)]
for j in range(10):
    for i in range(9):
        X1 = X[:,l2[:i+1]]
        Xs = torch.Tensor(X1)
        Ys = torch.Tensor(np.array(Ys))
        loss_list=[]
        model = net.TestNet(i+1, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for epoch in range(epochs):
            var_x = Variable(Xs, requires_grad=True)
            var_y = Variable(Ys, requires_grad=True)
        
            optimizer.zero_grad()
            out = model(var_x)
            loss = criterion(out, var_y)
        
            loss.backward()
            optimizer.step()
            #loss_list.append(loss.tolist())
        result[j].append(loss.tolist())
for  i in range(9):
    for j in range(10):
        print(result[j][i],end=" ")
    print("")
    
#torch.save(model.state_dict(), save_path)
