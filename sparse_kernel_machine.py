# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from matplotlib.pyplot import imshow
import idx2numpy

#%%
def rbf_func(X, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig2: scalar, bandwidth parameter for RBF function
    """
    sig2 = args['sig2']
    assert sig2 > 0
    
    dist = squareform(pdist(X, 'sqeuclidean'))
    return np.exp(-dist/(2*sig2))

def kernel(X, f, args):
    """
    X: numpy 2D array, the data matrix
    f: function, kernel function 
    args: dict object, other arguments for f
    """
    K = f(X, args)
    return K

def is_symmetric(X, tolr = 1E-8):
    """
    X: numpy 2D array
    """
    return np.max(np.abs(X - X.T)) < tolr

#%% 
class sparse_rkm:
    
    def __init__(self, eta = 0.5, eps = 10E-4, max_iter = 10 ** 4, max_tolr = 10E-4):
        """
        eta: scalar, regularization rate
        eps: scalar, tolerance in updating mu
        """
        assert eta > 0
        assert eps > 0
        
        self.eta = eta
        self.eps = eps
        self.max_iter = max_iter
        self.max_tolr = max_tolr
        
        self.H = None
        self.Lambda = None
        
    @staticmethod
    def solve_eigen(K, mu, s):
        coef_mat = K - np.diag(mu)
        assert is_symmetric(coef_mat)
        
        eigval, eigvec = np.linalg.eigh(coef_mat)
        max_ind = -1
        return eigval[max_ind], eigvec[:, max_ind]
    
    def check(self, X, pc):
        """
        Standard implementation of 
        X: numpy 2D array, the data matrix
        """
        n, s = X.shape
        K = 1/self.eta * kernel(X, rbf_func, {'sig2': 0.2})
        eigval, eigvec = np.linalg.eigh(K)
        
        self.H = eigvec[:, n - pc:]
        self.Lambda = eigval[n - pc:]
        
    
    def fit(self, X):
        """
        X: numpy 2D array, the data matrix
        """
        n, s = X.shape
        mu = np.ones((n, s))
        H = np.zeros((n, s))

        Lambda = np.zeros((1, s))
        K = 1/self.eta * kernel(X, rbf_func, {'sig2': 0.4})
        
        for l in range(self.max_iter):
            H_prev = H.copy()
            
            for j in range(s):
                Lambda[0, j], H[:, j] = self.solve_eigen(K, mu[:, j], s)
            
            if np.sum((H - H_prev) ** 2) < self.max_tolr:
                print('Convergence achieved.')
                self.H = H
                break
            
            mask = np.abs(H) > self.eps
            mu[:] = 1/self.eps ** 2
            mu[mask] = 1/H[mask] ** 2
            
            if l == self.max_iter - 1:
                print('Convergence fails.')
        
        self.H = H
        self.Lambda = Lambda

#%%
def test_code(X):
    srkm_model = sparse_rkm(max_iter = 1)
    srkm_model.check(X, pc = 100)
    return srkm_model

#%%
imagefile = r'./train-images-idx3-ubyte/train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

#%%
imagetrain = np.array(imagearray[:1000, :, :], dtype = np.float64)
imagedim = imagetrain.shape
imagetrain = imagetrain.reshape((imagedim[0], imagedim[1]*imagedim[2]))
imagetrain /= 255.0
output = test_code(imagetrain)
    