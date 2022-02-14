import torch
import torch.nn as nn
from math_custom import compute_kernel


# kernel PCA - (Suykens, 2017)
# Shallow case - level 1, implicit feature map
class KpcaStd(nn.Module):

    def __init__(self, input_dim, params):
        super().__init__()
        self.n_sample = input_dim[0]
        self.n_hidden = input_dim[1]

        self.K = None
        self.eta = params['eta']
        self.c_stab = params['c_stab']
        self.f = params['f']
        self.kernel_args = params['kernel_args']

        # initialization with Gaussian distribution
        scale = 10E-1
        self.H = nn.Parameter(scale * torch.randn(self.n_sample, self.n_hidden))
        self.inv_lambda_diag = nn.Parameter(scale * torch.randn(self.n_hidden))

    def forward(self, input_data):
        # compute kernel matrix
        self.K = compute_kernel(input_data, self.f, self.kernel_args)

        #  first part of loss, involving eigenvalues
        loss1 = 0
        for i in range(self.n_sample):
            e_j = torch.matmul(self.H.transpose(0, 1), self.K[:, i])
            loss1 += torch.dot(e_j, torch.mul(self.inv_lambda_diag, e_j))
        loss1 /= (-2 * self.eta ** 2)

        # second part of loss, involving matrix norm
        loss2 = torch.trace(torch.mm(torch.mm(self.H.transpose(0, 1), self.K), self.H))
        loss2 /= (2 * self.eta)

        # total loss
        loss_stab = (loss1 + loss2) + self.c_stab / 2 * torch.square(loss1 + loss2)
        return loss_stab
