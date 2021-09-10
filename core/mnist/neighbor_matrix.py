import numpy as np
import utils


def neighbormat(mat_size):
    matlist=[]
    mat=np.zeros((mat_size,mat_size), dtype=bool)
    for window_size in range(1, mat_size+1):
        for lineindex in range(mat_size - window_size + 1):
            for rowindex in range(mat_size - window_size + 1):
                mat[lineindex:lineindex+window_size,rowindex:rowindex+window_size] = 1
                print(mat)
                matlist.append(mat)
                mat=np.zeros((mat_size,mat_size), dtype=bool)
    return matlist


def neighbormat_v(mat_size):
    vec_list = []
    mat = np.zeros((mat_size, mat_size), dtype=bool)
    for window_size in range(1, mat_size + 1):
        for lineindex in range(mat_size - window_size + 1):
            for rowindex in range(mat_size - window_size + 1):
                mat[lineindex:lineindex + window_size, rowindex:rowindex + window_size] = 1
                vec_list.append(utils.mat_to_array(mat))
                mat = np.zeros((mat_size, mat_size), dtype=bool)
    return vec_list
