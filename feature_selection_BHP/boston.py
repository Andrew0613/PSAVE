import numpy as np
import torch
import pandas as pd
from d_calculator import DCalculatorPreMem
from v_calculator import VCalculator
import net
import find_fc
import psave
import w_with_fc
import time


feature_num = 10
save_path = "./model/test_net_10.pt"
met_path = "./metric/psave_10.txt"
torch.set_default_tensor_type(torch.FloatTensor)

result_csv = list()

# read data
dataset = pd.read_csv("./data/boston.csv")
for i in range(1):
    # prepare model
    start = time.perf_counter()
    model = net.TestNet(feature_num, 1)
    model.load_state_dict(torch.load(save_path))
    Xs = dataset.iloc[:, 0:feature_num]
    Ys = dataset.iloc[:, 13]
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    
    batch_size = 64
    calc_v = VCalculator(Xs, Ys, model, 'mse', batch_size)
    F = find_fc.find_fc(feature_num, calc_v)
    
    F = list()
    for i in range(2**9):
        boo = list()
        f_str = bin(i)[2:]
        for j in f_str:
            if(j == '0'):
                boo.append(False)
            else:
                boo.append(True)
        if(len(f_str)<10):
            for k in range(10-len(f_str)):
                boo.append(False)
        if boo[1]==True:
            F.append(np.array(boo))
    
    # loss 1: mse
    # loss 2: cross entropy
    calc_div = DCalculatorPreMem(Xs, Ys, model, F, 'mse', batch_size)
    w = w_with_fc.get_w_with_fc(feature_num, F, tp=True)  # np.ones(feature_num)
    res = psave.psave_whole_pre_mem(feature_num, w, F, calc_div, tp=True)
    np.savetxt(met_path, res)
    print("第"+str(i)+"轮")
    print(time.perf_counter()-start)
    result_csv.append(res)
ccsv = np.array(result_csv)
np.savetxt('data.csv',ccsv,delimiter=",")
