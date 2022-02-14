# kernel PCA - (Suykens, 2017)
# Shallow case - level 1, implicit feature map
# Direct eigendecomposition stated in (3.27)
class KpcaStdEigDecomp:

    def __init__(self, input_dim):
        self.n_sample = input_dim[0]
        self.s = input_dim[1]

        self.kernel_matrix = None

    def fit(self):
        