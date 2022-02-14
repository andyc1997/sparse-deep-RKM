import torch
from torch.nn.functional import pairwise_distance


# Typical kernel functions (Scholkopf & Smola, 2001)
def rbf_func(x, y, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig2: scalar, bandwidth parameter for RBF
    """
    sig2 = args['sig2']
    assert sig2 > 0

    n = x.shape[0]
    m = y.shape[0]
    dist_matrix = torch.zeros((m, n))

    for i in range(m):
        dist_matrix[i, :] = pairwise_distance(x, y[i, :], p=2)

    return torch.exp(- torch.square(dist_matrix) / 2 / sig2)


def laplace_func(x, y, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig: scalar, bandwidth parameter for laplace function
    """
    sig = args['sig']
    assert sig > 0

    n = x.shape[0]
    m = y.shape[0]
    dist_matrix = torch.zeros((m, n))

    for i in range(m):
        dist_matrix[i, :] = pairwise_distance(x, y[i, :], p=2)

    return torch.exp(- dist_matrix / sig)


def compute_kernel(x, f, args):
    """
    kernel matrix
    """
    n = x.shape[0]
    K = f(x, x, args)
    K = (K + K.transpose(0, 1)) / 2  # Ensure symmetry
    J = torch.ones((n, n)) / n
    K = K - torch.mm(J, K) - torch.mm(K, J) + torch.mm(J, torch.mm(K, J))  # Centering trick
    return K


def compute_cross_kernel(x, y, f, args, train_kernel):
    """
    cross kernel matrix
    """
    k = f(x, y, args).flatten()
    return k
