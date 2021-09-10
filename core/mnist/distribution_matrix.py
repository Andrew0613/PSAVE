import numpy as np
import utils


def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return (n*factorial(n-1))

def normalmat(mu1,sigma1,mu2,sigma2,rou,size):  # Gaussian distribution function
    matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            matrix[i][j]=((2*np.pi*sigma1*sigma2* ((1-rou**2)**(1/2)) )**(-1)) * (np.exp( (-1/(2*(1-rou**2))) * (((i-mu1)**2/(sigma1**2))
                                        -((2*rou*(i-mu1)*(j-mu2))/(sigma1*sigma2))+  ((j-mu2)**2/(sigma2**2))) ))
    return matrix


def normalmat_v(mu1,sigma1,mu2,sigma2,rou,size):
    return utils.mat_to_array(normalmat(mu1, sigma1, mu2, sigma2, rou, size))


def poissonmat(lamda,size):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j]=lamda**(i)*np.exp(-lamda)/factorial(i)*lamda**(j)*np.exp(-lamda)/factorial(j)
    return matrix


def poissonmat_v(lamda,size):
    return utils.mat_to_array(poissonmat(lamda, size))


def unifoemmat(size):
    return np.ones((size,size))/(size*size)


def unifoemmat_v(size):
    return utils.mat_to_array(unifoemmat(size))


def biomat(size,p):
    matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            n=size
            matrix[i][j]=factorial(n)/(factorial(i)*factorial(n-i)) * (p**i) * (1-p)**(n-i)*factorial(n) / (factorial(j) * factorial(n - j)) * (p ** j) * (1 - p) ** (n - j)
    return matrix


def biomat_v(size,p):
    return utils.mat_to_array(biomat(size, p))
