# -*- coding: utf-8 -*-

import numpy as np
from math_custom import compute_kernel, rbf_func

# kernel PCA - (Suykens, 2017)
# Shallow case - level 1
class KpcaStd:

    def __init__(self, input_dim):
        self.n_sample = input_dim[0]
        self.s = input_dim[1]
        assert self.s <= self.n_sample
        self.param = dict()

    def loss(self):
        """
        loss function, stabilized version, referred to J_{deep, P_stab} in the paper
        """
        # get param
        hidden, inv_lambda_diag = self.param['hidden'], self.param['inv_lambda_diag']
        kernel, eta, c_stab = self.param['kernel'], self.param['eta'], self.param['c_stab']

        # update cost
        cost = 0
        for i in range(self.n_sample):
            e = (hidden.dot(kernel[:, i])) / eta
            cost += - e.dot(e * inv_lambda_diag) / 2

        cost += eta / 2 * np.trace(hidden.dot(kernel.dot(hidden.T)))
        return cost + c_stab / 2 * (cost ** 2)

    def grad(self):
        """
        the gradient of loss function
        """
        # get param
        hidden, inv_lambda_diag = self.param['hidden'], self.param['inv_lambda_diag']
        kernel, eta, c_stab = self.param['kernel'], self.param['eta'], self.param['c_stab']

        # evaluate gradient
        dinv_lambda = np.zeros((self.s, self.s))
        # dhidden =

        for i in range(self.n_sample):
            e = (hidden.dot(kernel[:, i])) / eta
            dinv_lambda += np.outer(e, e)
        dinv_lambda /= -2

        pass

    def param_initialize(self):
        """initialize parameters by sampling """
        var = 1E-5
        hidden = var * np.random.randn(self.s, self.n_sample)
        inv_lambda_diag = var * np.random.randn(self.s)
        return hidden, inv_lambda_diag

    def fit(self, input_data, eta, c_stab, sig2):
        self.param['eta'] = eta
        self.param['c_stab'] = c_stab
        self.param['kernel'] = compute_kernel(input_data, rbf_func, {'sig2': sig2})

        self.param['hidden'], self.param['inv_lambda_diag'] = self.param_initialize()
        loss = self.loss()
        return loss
