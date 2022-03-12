The folder `sparse_gen_rkm` contains MATLAB scripts for developing <i>Gen-RKM algorithm</i> (Pandey, 2021) with sparsity corporated in the latent variables, using an <i>implicit feature map</i>. The files are organized as follows:

<b>Utils:</b>
- `random_seeds.m` generates 5 random seeds for experimental uses.
- `build_kernel.m` generates a kernel matrix given a training dataset; RBF kernel and laplace kernel are available.
- `preimg_sparse_kernel_pca_rkm.m` reconstructs the preimage from eigenvectors given noisy input using fixed point iterations (Mika et al., 1998). It assumes radial basis function (RBF) kernel.
- `gen_toy_example.m` generates toy example of 3 Gaussian distributions discussed in Pandey (2021).
- `toy.mat` is the generated dataset.

<b>Sparse PCA algorithms:</b>
- `GPower.m` is the implementation of <i>generalized power method</i> for sparse PCA (Journee et al., 2010), scraped form the authors' website.
- `TPower.m` is the implementation of <i>truncated power method</i> for sparse PCA (Yuan & Zhang, 2013).
- `mm_sparse_eigen.m` is the implementation of <i>majorization-minorization (MM) algorithm</i> with l0 norm for sparse PCA (Song et al., 2014).
- `sparse_deflation.m` is the implementation of alternative <i>matrix deflation schemes</i> for sparse PCA discussed in Mackey (2008).
- `test_sparse_eigenvectors.mlx` is the livescript for testing all developed sparse PCA algorithms with numerical experiments proposed by Journee et al. (2010).

<b>Gen-RKM algorithms:</b>
- `singleview_skpca_rkm.m` is the implementation of <i>sparse Gen-RKM algorithm</i> with implicit feature map when only single data source is given.
- `multiview_skpca_rkm.m` is the implementation of <i>sparse Gen-RKM algorithm</i> with implicit feature map when two data source are given.
- `gen_latent.m` is the implementation of <i>generative kernel PCA</i> given hidden variables, number of cluster and number of samples to be generated. It first fits a <i>Gaussian mixture model</i> on the latent space representation, and randomly sample from it.
- `gen_new_x.m` is the implementation of <i>generative kernel PCA</i> with <i>kernel smooting method</i>.

<b>Disentangled representation learning:</b>
- `experiment_toy_example.mlx` visualizes the disentangled property when sparse PCA is used in the <i>Gen-RKM algorithm</i>.
