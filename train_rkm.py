import numpy as np
import idx2numpy
import matplotlib
from matplotlib import pyplot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import kpca_std
import kpca_std_eigdecomp
from math_custom import *


def main(input_data, test_data):
    sig2 = input_data.shape[0] * torch.mean(torch.var(input_data, dim=1))
    test_data += 0.5 * torch.randn(test_data.shape) * torch.max(test_data)

    # n_sample = input_data.shape[0]
    # n_working_set = 100
    # reduced_set_kernel = reduced_set.ReducedSetMethod(n_sample, n_working_set, max_iter=1000)
    # reduced_set_kernel.fit(torch.tensor(input_data), rbf_func, {'sig2': 0.5})

    input_dim = (input_data.shape[0], 32)
    params = {'eta': 1, 'c_stab': 0.5, 'f': rbf_func, 'kernel_args': {'sig2': 0.7 * sig2}}

    model = kpca_std_eigdecomp.KpcaStdEigDecomp(input_dim)
    model.fit(input_data, params)
    test_data_reconstruct = model.reconstruct(input_data, image_test)
    return test_data_reconstruct, model.v

    # model = kpca_std.KpcaStd(input_dim, params)
    #
    # optimizer = optim.Adam(model.parameters(), lr = 10)
    # scheduler = ExponentialLR(optimizer, gamma = 0.1)
    #
    # # train
    # epoch = 0
    # n_epochs = 100
    #
    # while epoch <= n_epochs:
    #     optimizer.zero_grad()
    #     loss = model(input_data)
    #     loss.backward()  # get gradients w.r.t to parameters
    #     optimizer.step()  # update parameters
    #
    #     if epoch % 50 == 0:
    #         scheduler.step()  # update parameters
    #
    #     if epoch % 10 == 0:
    #         print(f'epoch = {epoch} J_stab={loss.item(): .6}')
    #
    #     epoch += 1
    #
    # # show parameter
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

# MNIST dataset for debugging
image_file = r'./train-images-idx3-ubyte/train-images.idx3-ubyte'
image_array = idx2numpy.convert_from_file(image_file)

# flatten
np.random.seed(0)
idx_train = np.random.choice(image_array.shape[0], 1000)
idx_test = np.random.choice(np.setdiff1d(np.arange(image_array.shape[0]), idx_train), 30)

image_train = np.array(image_array[idx_train, :, :], dtype=np.float64)
image_test = np.array(image_array[idx_test, :, :], dtype=np.float64)

image_dim = image_train.shape
image_train = image_train.reshape((image_dim[0], image_dim[1] * image_dim[2]))
image_dim = image_test.shape
image_test = image_test.reshape((image_dim[0], image_dim[1] * image_dim[2]))

image_train = torch.tensor(image_train)
image_test = torch.tensor(image_test)
image_clean = image_test.clone()
reconstruct, hidden = main(image_train, image_test)

print(torch.corrcoef(hidden[:, image_dim[0] - 1 - np.arange(10)]))

# plot

# fig, ax = pyplot.subplots(3, image_test.shape[0])
# for i in range(image_test.shape[0]):
#     ax[0, i].imshow(reconstruct[i, :].reshape(28, 28),
#                     cmap='Greys_r', interpolation='none')
#     ax[1, i].imshow(image_clean[i, :].reshape(28, 28),
#                     cmap='Greys_r', interpolation='none')
#     ax[2, i].imshow(image_test[i, :].reshape(28, 28),
#                     cmap='Greys_r', interpolation='none')
#
# pyplot.show()
