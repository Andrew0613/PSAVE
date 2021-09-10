import utils
from v_calculator import VCalculator


class DCalculatorPreMem:
    """
    This is for computing coalition dividend.
    There's a simple pretreatment of v on all coalitions.
    There's a simple memorized search.
    """

    def __init__(self, X, Y, f, F, loss_func, batch_size):
        """
        :param X: input values
        :param Y: target values
        :param loss_func: loss function
        :param batch_size: batch size of data
        """
        calc_v = VCalculator(X, Y, f, loss_func, batch_size)
        self.__pre = []
        self.__dic = {}
        m = len(F)
        for i in range(m):
            self.__pre.append(calc_v(F[i]))
            self.__dic[utils.set_hash(F[i])] = i
            print(i)
        self.__mem = {}

    def __call__(self, S, FS):
        """
        :param S: desired coalitions
        :param FS: feasible sub coalitions
        :return:
        """
        s_hash = utils.set_hash(S)
        if s_hash in self.__mem.keys():
            return self.__mem[s_hash]
        result = self.__pre[self.__dic[utils.set_hash(S)]]  # -> v(S)
        for nS in FS:
            result -= self.__call__(nS, utils.get_next_FS(nS, FS))
        self.__mem[s_hash] = result
        return result

    def get_mem(self):
        return self.__mem
