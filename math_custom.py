import torch
from torch.nn.functional import pairwise_distance

# Typical kernel functions (Scholkopf & Smola, 2001)
def rbf_func(x, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig2: scalar, bandwidth parameter for RBF function
    """
    sig2 = args['sig2']
    assert sig2 > 0

    n = x.shape[0]
    dist_matrix = torch.zeros((n, n))

    for i in range(n):
        dist_matrix[i, :] = pairwise_distance(x, x[i, :], p = 2)

    return torch.exp(- torch.square(dist_matrix) / 2 / sig2)

def laplace_func(x, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig: scalar, bandwidth parameter for laplace function
    """
    sig = args['sig']
    assert sig > 0

    n = x.shape[0]
    dist_matrix = torch.zeros((n, n))

    for i in range(n):
        dist_matrix[i, :] = pairwise_distance(x, x[i, :], p = 2)

    return torch.exp(- dist_matrix / sig)

def compute_kernel(x, f, args):
    """
    Kernel matrix with custom function
    """
    kernel_matrix = f(x, args)
    return (kernel_matrix + kernel_matrix.transpose(0, 1)) / 2
