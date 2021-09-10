import psave
import distribution_matrix
from matplotlib import pyplot as plt
from d_calculator import DCalculatorPreMem
import torch
import torchvision.datasets as dsets
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import find_fc


batch_size = 32
use_num = 128
model_path = "./model/mnist.pt"
save_path = "metric/psave_gau.txt"
device = torch.device('cuda', 0)

image_size = 28
width = 1
sp_num = image_size // width

F = find_fc.find_fc()
print(len(F))
w = distribution_matrix.normalmat_v(sp_num//2, 0.5, sp_num//2, 0.5, 0, sp_num)

# Load Dataset
# Load train set
train = dsets.MNIST('data', train=True, download=False)
imgs = train.data.reshape(-1, 784) / 255.0
labels = train.targets

# Shuffle and split into train and val
inds = torch.randperm(len(train))
imgs = imgs[inds]
labels = labels[inds]
val, Y_val = imgs[:6000], labels[:6000]
train, Y_train = imgs[6000:], labels[6000:]

# Load test set
test = dsets.MNIST('data', train=False, download=False)
test, Y_test = test.data.reshape(-1, 784) / 255.0, test.targets

# Move test data to numpy
test_np = test.cpu().data.numpy()
Y_test_np = Y_test.cpu().data.numpy()


model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ELU(),
    nn.Linear(256, 256),
    nn.ELU(),
    nn.Linear(256, 10)).to(device)

# Training parameters
lr = 1e-3
mbsize = 64
max_nepochs = 250
loss_fn = nn.CrossEntropyLoss()
lookback = 5
verbose = False

# Move to GPU
train = train.to(device)
val = val.to(device)
test = test.to(device)
Y_train = Y_train.to(device)
Y_val = Y_val.to(device)
Y_test = Y_test.to(device)

# Data loader
train_set = TensorDataset(train, Y_train)
train_loader = DataLoader(train_set, batch_size=mbsize, shuffle=True)

# Setup
optimizer = optim.Adam(model.parameters(), lr=lr)
min_criterion = np.inf
min_epoch = 0

# Train
for epoch in range(max_nepochs):
    for x, y in train_loader:
        # Move to device.
        x = x.to(device=device)
        y = y.to(device=device)

        # Take gradient step.
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        model.zero_grad()

    # Check progress.
    with torch.no_grad():
        # Calculate validation loss.
        val_loss = loss_fn(model(val), Y_val).item()
        if verbose:
            print('{}Epoch = {}{}'.format('-' * 10, epoch + 1, '-' * 10))
            print('Val loss = {:.4f}'.format(val_loss))

        # Check convergence criterion.
        if val_loss < min_criterion:
            min_criterion = val_loss
            min_epoch = epoch
            best_model = deepcopy(model)
        elif (epoch - min_epoch) == lookback:
            if verbose:
                print('Stopping early')
            break

# Keep best model
model = best_model
torch.save(model.state_dict(), './model/mnist.pt')

# Add activation at output
model_activation = nn.Sequential(model, nn.Softmax(dim=1))

p = torch.tensor([torch.mean((Y_test == i).float()) for i in range(10)], device=device)
base_ce = loss_fn(torch.log(p.repeat(len(Y_test), 1)), Y_test)
ce = loss_fn(model(test), Y_test)

print('Base rate cross entropy = {:.4f}'.format(base_ce))
print('Model cross entropy = {:.4f}'.format(ce))

calc_div = DCalculatorPreMem(image_size, width, test_np[:use_num], Y_test_np[:use_num], model_activation, F, 'cross entropy', batch_size)
result = psave.psave_whole_pre_mem(sp_num * sp_num, w, F, calc_div)
np.savetxt(save_path, result)

display = -np.reshape(result, (sp_num, sp_num))
plt.imshow(display, cmap="seismic")
plt.show()
