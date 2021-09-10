import numpy as np
import torch
import pandas as pd
from d_calculator import DCalculatorPreMem
from v_calculator import VCalculator
import net
import find_fc
import psave
import w_with_fc


feature_num = 10
save_path = "./model/test_net_10.pt"
met_path = "./metric/psave_10.txt"
torch.set_default_tensor_type(torch.FloatTensor)

# read data
dataset = pd.read_csv("./data/boston.csv")

# prepare model
model = net.TestNet(feature_num, 1)
model.load_state_dict(torch.load(save_path))
Xs = dataset.iloc[:, 0:feature_num]
Ys = dataset.iloc[:, 13]
Xs = np.array(Xs)
Ys = np.array(Ys)

batch_size = 64
calc_v = VCalculator(Xs, Ys, model, 'mse', batch_size)
F = find_fc.find_fc(feature_num, calc_v)
calc_div = DCalculatorPreMem(Xs, Ys, model, F, 'mse', batch_size)
w = w_with_fc.get_w_with_fc(feature_num, F)
res = psave.psave_whole_pre_mem(feature_num, w, F, calc_div, tp=True)
np.savetxt(met_path, res)
print(res)
print(len(F))
