import numpy as np


def find_fc():
    F = []
    for sz in range(2, 30, 2):
        for i in range(0, 28, 2):
            if i + sz - 1 >= 28:
                continue
            for j in range(0, 28, 2):
                if j + sz - 1 >= 28:
                    continue
                g = np.zeros(28 * 28, dtype=bool)
                for u in range(sz):
                    for v in range(sz):
                        g[(i+u)*28 + (j+v)] = True
                F.append(g.copy())
    return F
