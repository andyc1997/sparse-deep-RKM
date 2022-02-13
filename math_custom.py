import numpy as np
from scipy.spatial.distance import pdist, squareform

# Typical kernel functions (Scholkopf & Smola, 2001)
def rbf_func(x, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig2: scalar, bandwidth parameter for RBF function
    """
    sig2 = args['sig2']
    assert sig2 > 0

    dist = squareform(pdist(x, 'sqeuclidean'))
    return np.exp(-dist / (2 * sig2))

def laplace_func(x, args):
    """
    d: scalar, the squared euclidean distance between two vectors
    sig: scalar, bandwidth parameter for laplace function
    """
    sig = args['sig']
    assert sig > 0
    dist = squareform(pdist(x, 'sqeuclidean'))
    return np.exp(-dist / sig)

def compute_kernel(x, f, args):
    """
    Kernel matrix with custom function
    """
    kernel_matrix = f(x, args)
    return (kernel_matrix + kernel_matrix.T) / 2
