import numpy as np


def random_select(size, mrk):
    rev_mrk = (~mrk).astype(float)
    total = rev_mrk.sum()
    rev_mrk /= total
    return np.random.choice(size, p=rev_mrk)


def try_to_expand(size, cur_gp, mrk, calc_v):
    bs = calc_v(cur_gp)
    for i in range(size):
        if mrk[i]:
            continue
        cur_gp[i] = True
        if calc_v(cur_gp) >= bs:
            return i
        cur_gp[i] = False
    return None


def find_fc(size, calc_v):
    F = []
    tot_num = 0
    mrk = np.zeros(size, dtype=bool)
    while tot_num < size:
        start = random_select(size, mrk)
        print(start, end="#")
        cur_num = 1
        cur_gp = np.zeros(size, dtype=bool)
        cur_gp[start] = mrk[start] = True
        while True:
            F.append(cur_gp.copy())
            print(cur_num)
            nxt = try_to_expand(size, cur_gp, mrk, calc_v)
            if nxt is None:
                break
            cur_gp[nxt] = mrk[nxt] = True
            cur_num += 1
        tot_num += cur_num
    F.append(np.ones(size, dtype=bool))
    return F
