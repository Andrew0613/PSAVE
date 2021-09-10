from sage import utils
from imputers import MarginalImputer
import numpy as np


class VCalculator:
    """
    This is for computing value function.
    """
    def __init__(self, image_size, width, X, Y, f, loss_func, batch_size):
        """
        :param image_size: the image_size
        :param width: the super pixel width
        :param X: input values
        :param Y: target values
        :param f: the model
        :param loss_func: loss function
        """
        self.__X = X
        self.__Y = Y
        self.__f = f
        self.__loss_func = loss_func
        self.__batch_size = batch_size
        self.imputer = MarginalImputer(f, X)
        self.loss_fn = utils.get_loss(self.__loss_func, reduction='none')
        self.__epochs = 10
        self.__avg = 0
        S = np.zeros(image_size * image_size, dtype=bool)
        N = len(self.__X)
        for epoch in range(self.__epochs):
            mb = np.random.choice(N, self.__batch_size)
            x = self.__X[mb]
            y = self.__Y[mb]
            y_hat = self.imputer(x, S)
            self.__avg += np.mean(self.loss_fn(y_hat, y), axis=0)
        self.__avg /= self.__epochs
        print(self.__avg)
        print("Inside imputer init"+str(np.shape(self.__X)))

    def __call__(self, S):
        N = len(self.__X)
        loss = 0
        for epoch in range(self.__epochs):
            mb = np.random.choice(N, self.__batch_size)
            x = self.__X[mb]
            y = self.__Y[mb]
            y_hat = self.imputer(x, S)
            loss += np.mean(self.loss_fn(y_hat, y),axis=0)
        return self.__avg - loss/self.__epochs
