from math_custom import *
import numpy as np

class ReducedSetMethod:

    def __init__(self, n_sample, n_working):
        assert n_sample >= n_working

        self.n_sample = n_sample
        self.n_working = n_working
        self.idx_working_set = np.random.choice(n_sample, n_working)
        self.kernel_matrix = None

    def renyi_entropy(self):
        return - np.log(np.sum(self.kernel_matrix) / self.n_working ** 2)

    def fit(self, input_data, args):
        assert input_data.shape[0] == self.n_sample
        assert len(input_data.shape) == 2

        subset_data = input_data[self.idx_working_set, :]
        self.kernel_matrix = compute_kernel(subset_data, rbf_func, args)
        print(self.renyi_entropy())

