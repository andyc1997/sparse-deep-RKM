from math_custom import *
import random
import numpy as np
import torch


# Fixed-size method: using quadratic Renyi entropy - (Suykens et al., 2002)
class ReducedSetMethod:

    def __init__(self, n_sample, n_working, max_iter=10E3):
        assert n_sample >= n_working
        assert n_working > 0

        self.n_sample = n_sample
        self.n_working = n_working
        self.max_iter = int(max_iter)
        self.K = None
        self.entropy_trace = []  # store entropy history

        # initialize training set and working set
        self.idx_working_set = set(np.random.choice(n_sample, size=n_working, replace=False))
        self.training_set = set(np.arange(0, n_sample)) - self.idx_working_set

    def _renyi_entropy(self, kernel_matrix):
        """
        compute the quadratic Renyi entropy given a kernel matrix
        """
        return - float(torch.log(torch.sum(kernel_matrix) / self.n_working ** 2))

    def fit(self, input_data, f, args):
        assert input_data.shape[0] == self.n_sample
        assert len(input_data.shape) == 2

        subset_data = input_data[list(self.idx_working_set), :]
        self.K = compute_kernel(subset_data, f, args)
        entropy = self._renyi_entropy(self.K)
        self.entropy_trace.append(entropy)

        for _ in range(self.max_iter):
            # sample indices
            idx_ws_select = random.sample(self.idx_working_set, k=1)[0]
            idx_ts_select = random.sample(self.training_set, k=1)[0]

            # compute new Renyi entropy
            self.idx_working_set.add(idx_ts_select)
            self.idx_working_set.remove(idx_ws_select)

            new_K = compute_kernel(input_data[list(self.idx_working_set), :], f, args)
            new_entropy = self._renyi_entropy(new_K)

            # reject
            if new_entropy <= entropy:
                self.idx_working_set.remove(idx_ts_select)
                self.idx_working_set.add(idx_ws_select)
                self.entropy_trace.append(entropy)
                continue

            # accept
            self.K = new_K
            entropy = new_entropy
            self.entropy_trace.append(entropy)
