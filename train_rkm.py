import numpy as np
import idx2numpy
import matplotlib
from matplotlib import pyplot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import kpca_std
from math_custom import *

def main(input_data):
    input_data /= 255.0

    # n_sample = input_data.shape[0]
    # n_working_set = 100
    # reduced_set_kernel = reduced_set.ReducedSetMethod(n_sample, n_working_set, max_iter=1000)
    # reduced_set_kernel.fit(torch.tensor(input_data), rbf_func, {'sig2': 0.5})

    input_dim = (input_data.shape[0], 50)
    params = {'eta': 1, 'c_stab': 0.5, 'f': rbf_func, 'kernel_args': {'sig2': 0.5}}
    model = kpca_std.KpcaStd(input_dim, params)

    optimizer = optim.Adam(model.parameters(), lr = 10)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)

    # train
    epoch = 0
    n_epochs = 100

    while epoch <= n_epochs:
        optimizer.zero_grad()
        loss = model(input_data)
        loss.backward()  # get gradients w.r.t to parameters
        optimizer.step()  # update parameters

        if epoch % 50 == 0:
            scheduler.step()  # update parameters

        if epoch % 10 == 0:
            print(f'epoch = {epoch} J_stab={loss.item(): .6}')

        epoch += 1

    # show parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

# MNIST dataset for debugging
image_file = r'./train-images-idx3-ubyte/train-images.idx3-ubyte'
image_array = idx2numpy.convert_from_file(image_file)

# flatten
idx_train = np.random.choice(image_array.shape[0], 1000)
image_train = np.array(idx_train, dtype=np.float64)
image_dim = image_train.shape
image_train = image_train.reshape((image_dim[0], image_dim[1] * image_dim[2]))

image_train = torch.tensor(image_train)
main(image_train)
