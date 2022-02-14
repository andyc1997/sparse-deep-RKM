from math_custom import *
import numpy as np
import torch.linalg
import torch.nn.functional

# kernel PCA - (Suykens, 2017)
# Shallow case - level 1, implicit feature map
# Direct eigendecomposition stated in (3.27)

class KpcaStdEigDecomp:

    def __init__(self, input_dim):
        self.n_sample = input_dim[0]
        self.n_hidden = input_dim[1]
        self.K = None
        self.params = None

        self.e = None
        self.v = None

    def fit(self, input_data, params):
        self.params = params
        self.K = compute_kernel(input_data, params['f'], params['kernel_args'])
        self.e, self.v = torch.linalg.eigh(self.K / params['eta'])
        self.v = torch.nn.functional.normalize(self.v, dim=0)  # normalize eigenvector

    def preimage_approx(self, gamma, input_data, x0):
        z_old = x0.reshape(1, x0.shape[0])
        kernel_args = self.params['kernel_args']

        max_iter = 1000
        rerun = 0
        i = 0

        while i < max_iter:
            k = rbf_func(input_data, z_old, kernel_args)
            temp = (gamma * k).double()
            z = torch.matmul(input_data.transpose(0, 1), temp.flatten())
            z /= torch.sum(temp)
            z = z.reshape(1, z.shape[0])

            if rerun >= 3:
                print('Fail to converge!')
                return z

            if torch.any(torch.isnan(z)):
                i = 0
                z_old = torch.randn(1, x0.shape[0]) + torch.mean(input_data) * torch.sqrt(torch.var(input_data))
                rerun += 1
                continue

            if torch.norm(z - z_old) < 10E-6:
                print('converge!')
                return z

            z_old = z
            i += 1

        return z

    def reconstruct(self, input_data, test_data):
        idx_topk_pc = self.n_sample - np.arange(self.n_hidden) - 1
        e = self.e[idx_topk_pc]
        v = self.v[:, idx_topk_pc]

        test_preimage = torch.zeros(test_data.shape)
        f = self.params['f']
        kernel_args = self.params['kernel_args']

        n_test_sample = test_data.shape[0]
        centering = torch.eye(self.n_sample) - torch.ones((self.n_sample, self.n_sample)) / self.n_sample

        for i in range(n_test_sample):
            kx = compute_cross_kernel(input_data, test_data[i].reshape(1, test_data.shape[1]), f, kernel_args, self.K)
            kx_tilde = torch.matmul(centering, kx - torch.matmul(self.K, torch.ones(self.n_sample)) / self.n_sample)

            beta = torch.matmul(v.transpose(0, 1), kx_tilde) / torch.sqrt(e)
            gamma = torch.matmul(v, beta).flatten()
            gamma += (1 - torch.sum(gamma)) / self.n_sample

            test_preimage[i, :] = self.preimage_approx(gamma, input_data, test_data[i])

        return test_preimage

