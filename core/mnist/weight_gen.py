import distribution_matrix
from matplotlib import pyplot as plt
import torch
import numpy as np


batch_size = 32
use_num = 128
model_path = "./model/mnist.pt"
save_path = "./metric/weight.txt"
device = torch.device('cuda', 0)

image_size = 28
width = 1
sp_num = image_size // width

w = distribution_matrix.normalmat_v(sp_num//2, 0.5, sp_num//2, 0.5, 0, sp_num)

result = w
np.savetxt(save_path, result)

display = -np.reshape(result, (sp_num, sp_num))
plt.imshow(display, cmap="seismic")
plt.show()
