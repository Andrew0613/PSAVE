import numpy as np
import torch
import pandas as pd
import net
import sage


feature_num = 10
save_path = "./model/test_net_10.pt"
met_path = "./metric/psave_10.txt"
torch.set_default_tensor_type(torch.FloatTensor)

result_csv = list()

dataset = pd.read_csv("./data/boston.csv")

feature_names = [i for i in range(10)]

model = net.TestNet(feature_num, 1)
model.load_state_dict(torch.load(save_path))
# Xs = dataset.iloc[:, 0:feature_num]
Ys = dataset.iloc[:, 13]
X = dataset.iloc[:, 0:feature_num]
X = np.array(X)
# Xs = np.array(Xs)[:100]

Ys = np.array(Ys)[:500] # 506
Xs = np.ones([len(Ys),10])
for i in range(len(Ys)):
    Xs[i][0] = X[i][0]
    Xs[i][1] = X[i][1]
    Xs[i][2] = X[i][2]
    Xs[i][3] = X[i][3]
    Xs[i][4] = X[i][4]
    Xs[i][5] = X[i][5]
    Xs[i][6] = X[i][6]
    Xs[i][7] = X[i][7]
    Xs[i][8] = X[i][8]
    # Xs[i][9] = X[i][9]

result = []
for i in range(1):
    # Set up an imputer to handle missing features
    imputer = sage.MarginalImputer(model, Xs)
    
    # Set up an estimator
    estimator = sage.PermutationEstimator(imputer, 'mse')
    
    # Calculate SAGE values
    sage_values = estimator(Xs)
    sage_values.plot(feature_names)
    result.append(sage_values.values)
for j in range(10):
    for i in range(20):
        print(result[j][i],end=" ")
    print("")
