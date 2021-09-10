import operator as op


def set_eq(T, S):
    """
    Check whether T = S.
    :param T:
    :param S:
    :return:
    """
    if not op.eq(T.shape, S.shape):
        return False
    return (S^T).sum() == 0


def is_subset(T, S):
    """
    Check whether T is subset of S.
    :param T:
    :param S:
    :return:
    """
    if not op.eq(T.shape, S.shape):
        return False
    return (S|T == S).sum() == S.size


def is_in_set(x, S):
    """
    Check whether x is in set S. Assume S to be a 1-d numpy.ndarray.
    :param x:
    :param S:
    :return:
    """
    if x >= S.shape[0]:
        return False
    return S[x]


def get_next_FS(S, FS):
    """
    :param S:
    :param FS: F(S) \ S
    :return: the strict feasible sub coalitions of S under FS,
             represented as a list object, of which the elements
             are all numpy.ndarray.
    """
    result = []
    for nS in FS:
        if is_subset(nS, S) and not set_eq(nS, S):
            result.append(nS)
    return result


def set_hash(S):
    s_hash, m = 0, len(S)
    for i in range(m):
        if S[i]:
            s_hash = s_hash * 2 + 1
        else:
            s_hash = s_hash * 2
    return s_hash


def mat_to_array(mat_2d):
    return mat_2d.reshape(-1)


def array_to_mat(arr, shape):
    return arr.reshape(shape)
