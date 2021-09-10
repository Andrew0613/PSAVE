import utils
import numpy as np
from tqdm import tqdm


def psave_whole_pre_mem(feature_num, w, F, calc_div, tp=False):
    """
    :param feature_num: the number of features
    :param w: weight vector, 1-d numpy.ndarray
    :param F: feasible coalitions, a list object
    :param tp: the calculating type, if type is True, ignore the greatest coalition
    :return: the weighted SAGE on feasible coalitions
    """

    result = np.zeros(feature_num)
    calc_div(F[-1], utils.get_next_FS(F[-1], F))
    s_mem = calc_div.get_mem()
    # F[-1] denotes the greatest coalition.
    for S in tqdm(F):
        if tp and S.sum() == feature_num:
            continue
        w_sum = w[S].sum()
        d_S = s_mem[utils.set_hash(S)]
        for i in range(feature_num):
            if S[i]:
                result[i] += w[i] / w_sum * d_S
    print(s_mem)
    return result
