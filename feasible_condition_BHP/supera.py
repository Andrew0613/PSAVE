import numpy as np
import torch
import pandas as pd
from v_calculator import VCalculator
import net
import find_fc
import utils


feature_num = 10
save_path = "/Users/puyuandong613/Downloads/DNN的可解释性/boston_s/model/test_net_10.pt"
torch.set_default_tensor_type(torch.FloatTensor)

# read data
dataset = pd.read_csv("/Users/puyuandong613/Downloads/DNN的可解释性/boston_s/data/boston.csv")

# prepare model
model = net.TestNet(feature_num, 1)
model.load_state_dict(torch.load(save_path))
Xs = dataset.iloc[:, 0:feature_num]
Ys = dataset.iloc[:, 13]
Xs = np.array(Xs)
Ys = np.array(Ys)

batch_size = 64
calc_v = VCalculator(Xs, Ys, model, 'mse', batch_size)
vs = {}
full_coalition = []
feasible_coalition = []
full_tot = 0
feasible_tot = 0
epochs = 20
for num in range(epochs):
    print("# " + str(num) + ' #')
    F = utils.get_full_set(feature_num)
    epoch = 0
    for S in F:
        print(epoch)
        epoch += 1
        s_hash = utils.set_hash(S)
        vs[s_hash] = calc_v(S)
    cnt, tot = 0, 0
    for S in F:
        for T in F:
            if utils.set_eq(T, S):
                continue
            tot += 1
            s_hash = utils.set_hash(S)
            t_hash = utils.set_hash(T)
            s_cup_t_hash = utils.set_hash(S|T)
            v_s = vs.get(s_hash, None)
            v_t = vs.get(t_hash, None)
            v_s_cup_t = vs.get(s_cup_t_hash, None)
            if v_s is None or v_t is None or v_s_cup_t is None:
                cnt += 1
                continue
            if v_s + v_t < v_s_cup_t:
                cnt += 1
                continue
    print("full coalitions: {:.4f} {} {}".format(cnt / tot, cnt, tot))
    full_coalition.append(cnt/tot)
    full_tot += cnt/tot
    F = find_fc.find_fc_v(feature_num, vs)
    vss = {}
    epoch = 0
    for S in F:
        print(epoch)
        epoch += 1
        s_hash = utils.set_hash(S)
        vss[s_hash] = vs[s_hash]
    ncnt, tot = 0, 0
    for S in F:
        for T in F:
            if utils.set_eq(T, S):
                continue
            tot += 1
            s_hash = utils.set_hash(S)
            t_hash = utils.set_hash(T)
            s_cup_t_hash = utils.set_hash(S|T)
            v_s = vss.get(s_hash, None)
            v_t = vss.get(t_hash, None)
            v_s_cup_t = vss.get(s_cup_t_hash, None)
            if v_s is None or v_t is None or v_s_cup_t is None:
                ncnt += 1
                continue
            if v_s + v_t < v_s_cup_t:
                ncnt += 1
                continue
    print("feasible coalitions: {:.4f} {} {}".format(ncnt / tot, ncnt, tot))
    feasible_coalition.append(ncnt/tot)
    feasible_tot += ncnt/tot
    print("feasible coalitions (recall): {:.4f}".format(ncnt / cnt))
with open ('/Users/puyuandong613/Downloads/DNN的可解释性/boston_s/superfull_coalition_a.txt', 'w') as f:
    f.write(str(full_coalition))
with open ('/Users/puyuandong613/Downloads/DNN的可解释性/boston_s/super/feasible_coalition_a.txt', 'w') as f:
    f.write(str(feasible_coalition))
print("average of full:", full_tot/epochs)
print("average of feasible:", feasible_tot/epochs)
