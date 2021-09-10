import numpy as np


def get_w_with_fc(feature_num, F ,tp=False):
    w = np.zeros(feature_num, dtype=float)
    for S in F:
        if tp == True and S.sum() == feature_num:
            continue
        for i in range(feature_num):
            if S[i]:
                w[i] += 1
    return w
